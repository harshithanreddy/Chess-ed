"""Download a Lichess monthly dump into data/raw/ (git-ignored). Resumable."""
from __future__ import annotations

import time
import urllib.request
from urllib.error import HTTPError

from .paths import RAW_DIR, rel

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def download(month: str, dest_dir=RAW_DIR, log_every_seconds: float = 15.0):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"lichess_db_standard_rated_{month}.pgn.zst"
    if dest.exists():
        print(f"[skip] {rel(dest)} exists")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(URL.format(month=month))
    if start:
        req.add_header("Range", f"bytes={start}-")

    try:
        r = urllib.request.urlopen(req, timeout=60)
    except HTTPError as e:
        if e.code == 416 and start:  # .part already holds the whole file
            part.rename(dest)
            print(f"[done] {rel(dest)}")
            return dest
        raise

    with r:
        resumed = start > 0 and r.status == 206
        if start and not resumed:
            start = 0  # server ignored Range: restart from scratch
        total = start + int(r.headers.get("Content-Length", 0))
        done = start
        if resumed:
            print(f"  resuming at {done / 1e6:,.0f} MB")
        last = time.monotonic()
        with open(part, "ab" if resumed else "wb") as f:
            while block := r.read(1 << 20):
                f.write(block)
                done += len(block)
                now = time.monotonic()
                if total and now - last >= log_every_seconds:
                    print(f"  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB ({done / total:.0%})", flush=True)
                    last = now

    if total and done < total:
        raise OSError(f"incomplete download: {done} of {total} bytes; re-run to resume")
    part.rename(dest)
    print(f"[done] {rel(dest)}")
    return dest