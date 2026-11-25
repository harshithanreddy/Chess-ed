from __future__ import annotations

import argparse
from pathlib import Path

from .paths import RAW_DIR


def main(argv=None):
    p = argparse.ArgumentParser(prog="metalens")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download a monthly dump, e.g. 2016-01")
    d.add_argument("month")

    i = sub.add_parser("ingest", help="ingest a month (2016-01) or a file path")
    i.add_argument("target")
    i.add_argument("--workers", type=int, default=None)
    i.add_argument("--force", action="store_true")
    i.add_argument("--limit-chunks", type=int, default=None, help="dev: only first N chunks")

    v = sub.add_parser("validate", help="run data-quality checks on an ingested month")
    v.add_argument("month")
    v.add_argument("--memory-limit", default="6GB")
    v.add_argument("--sample-rows", type=int, default=100_000)

    r = sub.add_parser("run", help="Prefect pipeline: download -> ingest -> validate")
    r.add_argument("months", nargs="+", help="e.g. 2016-01 2016-02")
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--delete-raw", action="store_true", help="delete the .pgn.zst after a passing month")
    r.add_argument("--memory-limit", default="6GB")

    an = sub.add_parser("analyze", help="build dashboard aggregates (meta shift, matchmaking, time pressure)")
    an.add_argument("--speed", default="blitz")
    an.add_argument("--memory-limit", default="6GB")

    tw = sub.add_parser("train-win", help="train the win/draw/loss probability model")
    tw.add_argument("--speed", default="blitz")
    tw.add_argument("--month", default=None, help="restrict to one ingested month")
    tw.add_argument("--memory-limit", default="6GB")

    tr = sub.add_parser("train-retention", help="train the player retention model")
    tr.add_argument("feature_month", help="e.g. 2024-01 (label comes from the NEXT ingested month)")
    tr.add_argument("--label-month", default=None)
    tr.add_argument("--speed", default="blitz")
    tr.add_argument("--shap-sample", type=int, default=5000)
    tr.add_argument("--memory-limit", default="6GB")

    b = sub.add_parser("benchmark", help="parser vs python-chess")
    b.add_argument("--games", type=int, default=20_000)

    a = p.parse_args(argv)
    if a.cmd == "download":
        from .download import download
        download(a.month)
    elif a.cmd == "ingest":
        from .ingest import ingest_file
        path = Path(a.target)
        if not path.exists():  # treat as a month like 2016-01
            path = RAW_DIR / f"lichess_db_standard_rated_{a.target}.pgn.zst"
        ingest_file(path, workers=a.workers, force=a.force, limit_chunks=a.limit_chunks)
    elif a.cmd == "validate":
        from .quality import validate_month
        validate_month(a.month, sample_rows=a.sample_rows, memory_limit=a.memory_limit)
    elif a.cmd == "run":
        from .flow import monthly_pipeline
        monthly_pipeline(a.months, workers=a.workers, delete_raw=a.delete_raw,
                         memory_limit=a.memory_limit)
    elif a.cmd == "analyze":
        from .analytics.db import connect
        from .analytics.queries import build_all
        from .paths import AGG_DIR
        con = connect(memory_limit=a.memory_limit)
        build_all(con, AGG_DIR, speed=a.speed)
    elif a.cmd == "train-win":
        from .analytics.db import connect
        from .ml.win_model import train_and_evaluate
        con = connect(memory_limit=a.memory_limit)
        train_and_evaluate(con, speed=a.speed, month=a.month)
    elif a.cmd == "train-retention":
        from .analytics.db import connect
        from .ml.retention import train_and_evaluate as train_retention
        con = connect(memory_limit=a.memory_limit)
        train_retention(con, feature_month=a.feature_month, label_month=a.label_month, speed=a.speed,
                       shap_sample_size=a.shap_sample)
    elif a.cmd == "benchmark":
        from .benchmark import run
        run(a.games)


if __name__ == "__main__":
    main()
