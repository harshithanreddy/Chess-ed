"""Benchmark the custom parser against python-chess on the committed sample."""
from __future__ import annotations

import io
import json
import platform
import time

from .parser import parse_chunk
from .paths import BENCH_DIR, SAMPLE_DIR


def _timeit(fn) -> float:
    t = time.perf_counter()
    n = fn()
    return n / (time.perf_counter() - t)


def run(n_games: int = 20_000) -> dict:
    sample = (SAMPLE_DIR / "sample.pgn").read_text(encoding="utf-8")
    per = sum(1 for ln in sample.splitlines() if ln.startswith("[Event "))
    text = sample.rstrip("\n") + "\n\n"
    text = text * max(1, n_games // per)
    n = per * max(1, n_games // per)

    results = {"games": n, "python": platform.python_version(), "games_per_sec": {}}
    results["games_per_sec"]["metalens_parser_1proc"] = round(_timeit(lambda: len(parse_chunk(text)[0])), 1)

    try:
        import chess.pgn

        def full():
            h, c = io.StringIO(text), 0
            while chess.pgn.read_game(h) is not None:
                c += 1
            return c

        def headers_only():
            h, c = io.StringIO(text), 0
            while chess.pgn.read_headers(h) is not None:
                c += 1
                chess.pgn.skip_game(h)
            return c

        results["games_per_sec"]["python_chess_read_game"] = round(_timeit(full), 1)
        results["games_per_sec"]["python_chess_headers_only"] = round(_timeit(headers_only), 1)
    except ImportError:
        results["note"] = "python-chess not installed; run: pip install -e .[dev]"

    BENCH_DIR.mkdir(exist_ok=True)
    (BENCH_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results
