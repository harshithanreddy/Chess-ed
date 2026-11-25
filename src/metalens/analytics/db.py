"""Shared DuckDB connection over all ingested months (hive-partitioned Parquet)."""
from __future__ import annotations

from pathlib import Path

from ..paths import PROCESSED_DIR, rel


def _q(path: Path) -> str:
    return Path(path).as_posix().replace("'", "''")


def connect(root: Path = PROCESSED_DIR, memory_limit: str = "6GB"):
    import duckdb

    months = sorted(p.name.split("=")[1] for p in Path(root).glob("month=*") if (p / "_SUCCESS").exists())
    if not months:
        raise FileNotFoundError(f"No ingested (passed) months found under {rel(root)}. Run 'metalens run <month>' first.")

    tmp = Path(root) / ".duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{_q(tmp)}'")
    glob = _q(Path(root) / "month=*" / "games" / "*" / "*.parquet")
    # month + speed come from the hive-partitioned folder names, not file contents
    con.execute(f"""
        CREATE VIEW games AS
        SELECT *, regexp_extract(filename, 'month=([0-9]{{4}}-[0-9]{{2}})', 1) AS month
        FROM read_parquet('{glob}', hive_partitioning=true, filename=true)
        -- drop games aborted before move 1 (see quality.py zero_move_games warning):
        -- they have no opening/meta signal and would skew ply/duration stats
        WHERE ply_count > 0
    """)
    con.execute("CREATE VIEW months AS SELECT DISTINCT month FROM games ORDER BY 1")
    print(f"[analytics] connected: months={months}")
    return con
