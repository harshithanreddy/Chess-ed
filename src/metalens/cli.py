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
    elif a.cmd == "benchmark":
        from .benchmark import run
        run(a.games)


if __name__ == "__main__":
    main()
