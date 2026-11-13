"""Every path in the project is derived from the repo root.

No absolute paths anywhere: clone the repo anywhere, on any machine, and it works.
Raw/processed data live INSIDE the repo folder but are git-ignored, so deleting
them never touches anything that is committed or deployed.
"""
from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repo root (no pyproject.toml found upward).")


REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # downloaded .pgn.zst   (git-ignored, deletable)
PROCESSED_DIR = DATA_DIR / "processed"  # full parquet output   (git-ignored, deletable)
SAMPLE_DIR = DATA_DIR / "sample"        # tiny sample           (committed)
AGG_DIR = DATA_DIR / "aggregated"       # dashboard inputs      (committed)
BENCH_DIR = REPO_ROOT / "benchmarks"    # benchmark results     (committed)


def rel(path: Path) -> str:
    """Repo-relative string, for logs and manifests (never leaks machine paths)."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).name
