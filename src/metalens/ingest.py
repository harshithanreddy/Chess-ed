"""Streaming, parallel ingestion: .pgn.zst -> partitioned Parquet.

* Never decompresses to disk; reads the zstd stream in large binary blocks.
* Main process only cuts blocks at game boundaries; workers parse + write Parquet.
* Bounded in-flight chunks => constant memory regardless of file size.
* Idempotent: a finished month has a _SUCCESS marker and is skipped.
* Output: data/processed/month=YYYY-MM/games/speed=<speed>/*.parquet
          data/processed/month=YYYY-MM/quarantine/*.parquet (malformed games)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import shutil
import threading
import time
from pathlib import Path

from .parser import parse_chunk
from .paths import PROCESSED_DIR, rel

CHUNK_BYTES = 16 * 1024 * 1024
_BOUNDARY = b"\n[Event "


def infer_month(path: Path) -> str:
    m = re.search(r"(\d{4}-\d{2})", Path(path).name)
    if not m:
        raise ValueError(f"Cannot infer YYYY-MM from filename: {Path(path).name}")
    return m.group(1)


def _open_binary(path: Path):
    if str(path).endswith(".zst"):
        import zstandard

        return zstandard.ZstdDecompressor().stream_reader(open(path, "rb"), read_across_frames=True)
    return open(path, "rb")


def iter_chunks(raw, chunk_bytes: int = CHUNK_BYTES):
    """Yield byte chunks that always end exactly at a game boundary."""
    carry = b""
    while True:
        block = raw.read(chunk_bytes)
        if not block:
            if carry.strip():
                yield carry
            return
        data = carry + block
        cut = data.rfind(_BOUNDARY)
        if cut == -1:
            carry = data
            continue
        yield data[: cut + 1]
        carry = data[cut + 1 :]


def _schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("game_id", pa.string()),
            ("played_at", pa.timestamp("s")),
            ("speed", pa.string()),
            ("rated", pa.bool_()),
            ("white", pa.string()),
            ("black", pa.string()),
            ("white_elo", pa.int16()),
            ("black_elo", pa.int16()),
            ("white_rating_diff", pa.int16()),
            ("black_rating_diff", pa.int16()),
            ("result", pa.string()),
            ("white_score", pa.float32()),
            ("eco", pa.string()),
            ("opening", pa.string()),
            ("base_seconds", pa.int32()),
            ("increment_seconds", pa.int32()),
            ("termination", pa.string()),
            ("ply_count", pa.int16()),
            ("has_eval", pa.bool_()),
            ("has_clock", pa.bool_()),
            ("moves", pa.string()),
            ("clocks", pa.list_(pa.int32())),
        ]
    )


def _work(args):
    idx, chunk, out_dir = args
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _schema()
    records, rejects = parse_chunk(chunk.decode("utf-8", errors="replace"))
    if records:
        table = pa.table({n: [r[n] for r in records] for n in schema.names}, schema=schema)
        pq.write_to_dataset(
            table,
            root_path=str(Path(out_dir) / "games"),
            partition_cols=["speed"],
            basename_template=f"chunk{idx:06d}-{{i}}.parquet",
            compression="zstd",
            existing_data_behavior="overwrite_or_ignore",
        )
    if rejects:
        qdir = Path(out_dir) / "quarantine"
        qdir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rejects), qdir / f"chunk{idx:06d}.parquet", compression="zstd")
    return len(records), len(rejects), len(chunk)


def ingest_file(path: Path, workers: int | None = None, force: bool = False,
                out_root: Path = PROCESSED_DIR, limit_chunks: int | None = None) -> dict:
    path = Path(path)
    month = infer_month(path)
    final = Path(out_root) / f"month={month}"
    if (final / "_SUCCESS").exists() and not force:
        print(f"[skip] {month} already ingested ({rel(final)}). Use --force to redo.")
        return json.loads((final / "_SUCCESS").read_text())

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    tmp = Path(out_root) / f".tmp-month={month}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    sem = threading.BoundedSemaphore(workers * 2)  # caps memory: at most 2 chunks/worker queued

    def gen(raw):
        for i, chunk in enumerate(iter_chunks(raw)):
            if limit_chunks is not None and i >= limit_chunks:
                return
            sem.acquire()
            yield (i, chunk, str(tmp))

    ok = bad = nbytes = 0
    t0 = time.perf_counter()
    with _open_binary(path) as raw, mp.get_context("spawn").Pool(workers) as pool:
        for n_ok, n_bad, n_bytes in pool.imap_unordered(_work, gen(raw)):
            sem.release()
            ok, bad, nbytes = ok + n_ok, bad + n_bad, nbytes + n_bytes
            if (ok + bad) % 200_000 < n_ok + n_bad:
                print(f"  {ok + bad:>12,} games  {(ok + bad) / (time.perf_counter() - t0):>9,.0f} games/s")
    secs = time.perf_counter() - t0

    manifest = {
        "month": month,
        "source": Path(path).name,
        "games_ok": ok,
        "games_quarantined": bad,
        "uncompressed_bytes": nbytes,
        "seconds": round(secs, 2),
        "games_per_second": round((ok + bad) / secs, 1) if secs else None,
        "workers": workers,
    }
    (tmp / "_SUCCESS").write_text(json.dumps(manifest, indent=2))
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)  # atomic publish: a month is either complete or absent
    print(f"[done] {month}: {ok:,} ok, {bad:,} quarantined, {manifest['games_per_second']:,} games/s")
    return manifest
