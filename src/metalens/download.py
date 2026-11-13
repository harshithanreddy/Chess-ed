"""Download a Lichess monthly dump into data/raw/ (git-ignored)."""
from __future__ import annotations

import urllib.request

from .paths import RAW_DIR, rel

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def download(month: str, dest_dir=RAW_DIR):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"lichess_db_standard_rated_{month}.pgn.zst"
    if dest.exists():
        print(f"[skip] {rel(dest)} exists")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(URL.format(month=month)) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while block := r.read(1 << 20):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    part.rename(dest)
    print(f"\n[done] {rel(dest)}")
    return dest
