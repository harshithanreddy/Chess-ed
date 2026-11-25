"""Data-quality gate for an ingested month.

Two layers:
1. Full-month checks in DuckDB SQL (scales to 100M+ rows, spills to disk if needed).
2. A Pandera schema "contract" on a random sample (types, ranges, uniqueness).

Hard failures raise QualityError so the pipeline stops before bad data is analysed.
A JSON report is written next to the data AND to data/aggregated/quality/ (committed),
so the repo and dashboard can show data-quality results after raw data is deleted.
"""
from __future__ import annotations

import json
from pathlib import Path

from .paths import AGG_DIR, PROCESSED_DIR, rel

SPEEDS = ["ultrabullet", "bullet", "blitz", "rapid", "classical", "correspondence", "other"]
RESULTS = ["1-0", "0-1", "1/2-1/2"]
MAX_QUARANTINE_RATIO = 0.01  # >1% malformed games => something is wrong upstream

SAMPLE_COLUMNS = (
    "game_id, speed, result, white_score, white_elo, black_elo, base_seconds, "
    "increment_seconds, ply_count, rated, has_eval, has_clock"
)


class QualityError(RuntimeError):
    pass


def _q(path: Path) -> str:
    """Posix-style, SQL-escaped path literal (works for Windows paths too)."""
    return Path(path).as_posix().replace("'", "''")


def _checks(month: str) -> list[tuple[str, str, str]]:
    """(name, severity, SQL returning the number of violating rows)."""
    return [
        ("null_game_id", "error", "SELECT count(*) FROM games WHERE game_id IS NULL OR game_id = ''"),
        ("duplicate_game_ids", "error", "SELECT count(*) - count(DISTINCT game_id) FROM games"),
        ("invalid_result", "error",
         "SELECT count(*) FROM games WHERE result NOT IN ('1-0','0-1','1/2-1/2')"),
        ("score_result_mismatch", "error",
         ("SELECT count(*) FROM games WHERE white_score <> "
         "CASE result WHEN '1-0' THEN 1.0 WHEN '0-1' THEN 0.0 ELSE 0.5 END")),
        ("elo_out_of_range", "error",
         ("SELECT count(*) FROM games WHERE white_elo < 100 OR white_elo > 4000 "
         "OR black_elo < 100 OR black_elo > 4000")),
        ("negative_ply_count", "error", "SELECT count(*) FROM games WHERE ply_count < 0"),
        ("date_outside_month", "warn",
         f"SELECT count(*) FROM games WHERE strftime(played_at, '%Y-%m') <> '{month}'"),
        ("negative_clock_value", "warn", "SELECT count(*) FROM games WHERE list_min(clocks) < 0"),
        ("unknown_speed", "warn", "SELECT count(*) FROM games WHERE speed = 'other'"),
        ("zero_move_games", "warn", "SELECT count(*) FROM games WHERE ply_count = 0"),
    ]


def _contract():
    try:
        import pandera.pandas as pa
    except ImportError:  # older pandera
        import pandera as pa
    return pa.DataFrameSchema(
        {
            "game_id": pa.Column(str, nullable=False, unique=True),
            "speed": pa.Column(str, pa.Check.isin(SPEEDS)),
            "result": pa.Column(str, pa.Check.isin(RESULTS)),
            "white_score": pa.Column(float, pa.Check.isin([0.0, 0.5, 1.0]), coerce=True),
            "white_elo": pa.Column(float, pa.Check.in_range(100, 4000), nullable=True, coerce=True),
            "black_elo": pa.Column(float, pa.Check.in_range(100, 4000), nullable=True, coerce=True),
            "base_seconds": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
            "increment_seconds": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
            "ply_count": pa.Column(int, pa.Check.ge(0), coerce=True),
            "rated": pa.Column(bool, coerce=True),
            "has_eval": pa.Column(bool, coerce=True),
            "has_clock": pa.Column(bool, coerce=True),
        },
        strict=False,
    )


def validate_month(month: str, root: Path = PROCESSED_DIR, agg_dir: Path = AGG_DIR,
                   sample_rows: int = 100_000, memory_limit: str = "6GB",
                   raise_on_error: bool = True) -> dict:
    import duckdb
    from pandera.errors import SchemaErrors

    mdir = Path(root) / f"month={month}"
    success = mdir / "_SUCCESS"
    if not success.exists():
        raise FileNotFoundError(f"{rel(mdir)} has no _SUCCESS marker; run ingest first.")
    manifest = json.loads(success.read_text())

    tmp = Path(root) / ".duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{_q(tmp)}'")
    games_glob = _q(mdir / "games" / "*" / "*.parquet")
    con.execute(
        f"CREATE VIEW games AS SELECT * FROM read_parquet('{games_glob}', hive_partitioning=true)"
    )

    report: dict = {"month": month, "checks": [], "errors": [], "warnings": []}

    # --- integrity vs ingest manifest ---
    n_rows = con.execute("SELECT count(*) FROM games").fetchone()[0]
    report["rows"] = n_rows
    report["manifest_rows"] = manifest["games_ok"]
    if n_rows == 0:
        report["errors"].append("no_rows")
    if n_rows != manifest["games_ok"]:
        report["errors"].append(f"row_count_mismatch: parquet={n_rows} manifest={manifest['games_ok']}")

    # --- quarantine ratio ---
    q_bad, q_ok = manifest["games_quarantined"], manifest["games_ok"]
    ratio = q_bad / (q_bad + q_ok) if (q_bad + q_ok) else 0.0
    report["quarantine_ratio"] = round(ratio, 6)
    if ratio > MAX_QUARANTINE_RATIO:
        report["errors"].append(f"quarantine_ratio_too_high: {ratio:.4%}")
    qdir = mdir / "quarantine"
    if qdir.exists():
        rows = con.execute(
            f"SELECT split_part(reason, ':', 1) AS r, count(*) FROM "
            f"read_parquet('{_q(qdir / '*.parquet')}') GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
        report["quarantine_reasons"] = {r: n for r, n in rows}

    # --- full-month SQL checks ---
    for name, severity, sql in _checks(month):
        violations = con.execute(sql).fetchone()[0]
        report["checks"].append({"name": name, "severity": severity, "violations": violations})
        if violations:
            bucket = "errors" if severity == "error" else "warnings"
            report[bucket].append(f"{name}: {violations:,} rows")

    # --- Pandera contract on a random sample ---
    sample = con.execute(
        f"SELECT {SAMPLE_COLUMNS} FROM games USING SAMPLE reservoir({int(sample_rows)} ROWS)"
    ).df()
    report["sample_rows"] = len(sample)
    try:
        _contract().validate(sample, lazy=True)
        report["contract"] = "passed"
    except SchemaErrors as e:
        report["contract"] = "failed"
        report["contract_failures"] = e.failure_cases.head(20).astype(str).to_dict("records")
        report["errors"].append("pandera_contract_failed")
    con.close()

    report["status"] = "failed" if report["errors"] else "passed"
    text = json.dumps(report, indent=2, default=str)
    (mdir / "_quality.json").write_text(text)
    out = Path(agg_dir) / "quality"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{month}.json").write_text(text)  # committed: survives raw-data deletion

    print(f"[quality] {month}: {report['status']} ({n_rows:,} rows, "
          f"{len(report['errors'])} errors, {len(report['warnings'])} warnings)")
    if report["errors"] and raise_on_error:
        raise QualityError(f"{month} failed quality checks: {report['errors']}")
    return report
