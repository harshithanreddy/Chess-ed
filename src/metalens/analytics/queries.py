"""Analytics queries: meta shift, matchmaking fairness, time pressure.

Every query conditions on rating band where relevant (see proposal section 5:
raw win rates are confounded by rating, since stronger players pick different
openings). Each function returns a DuckDB relation .df() away from pandas, or
writes directly to Parquet for the dashboard via *_to_parquet().
"""
from __future__ import annotations

from pathlib import Path

RATING_BANDS = "CASE WHEN white_elo IS NULL THEN 'unrated/unknown' " \
    "WHEN white_elo < 1200 THEN '<1200' WHEN white_elo < 1600 THEN '1200-1599' " \
    "WHEN white_elo < 2000 THEN '1600-1999' WHEN white_elo < 2400 THEN '2000-2399' " \
    "ELSE '2400+' END"

MIN_GAMES_PER_OPENING = 200  # drop rare openings: their win rate is noise, not signal


def _bucket_expr(col: str, lo: float, hi: float, n: int) -> str:
    """Equal-width bucket index in [0, n-1], clamping outliers into the end buckets.

    Equivalent to width_bucket(col, lo, hi, n), which this DuckDB build doesn't have.
    """
    width = (hi - lo) / n
    clamped = f"LEAST(GREATEST({col}, {lo}), {hi} - 0.0001)"
    return f"CAST(FLOOR(({clamped} - {lo}) / {width}) AS INTEGER)"


def opening_meta_by_month(con, speed: str = "blitz", min_games: int = MIN_GAMES_PER_OPENING):
    """Opening popularity & white win rate, by month and rating band.

    Confound control: popularity and win rate are computed WITHIN each rating
    band, so a shift here can't just be "strong players becoming more common".
    """
    return con.execute(f"""
        WITH per_band AS (
            SELECT month, {RATING_BANDS} AS rating_band, eco, opening,
                   count(*) AS n_games,
                   avg(white_score) AS white_win_rate
            FROM games
            WHERE speed = '{speed}' AND rated AND opening IS NOT NULL
            GROUP BY 1, 2, 3, 4
            HAVING count(*) >= {min_games}
        ),
        band_totals AS (
            SELECT month, rating_band, sum(n_games) AS band_games
            FROM per_band GROUP BY 1, 2
        )
        SELECT p.month, p.rating_band, p.eco, p.opening, p.n_games,
               round(p.n_games * 1.0 / b.band_games, 5) AS pick_rate,
               round(p.white_win_rate, 4) AS white_win_rate
        FROM per_band p JOIN band_totals b USING (month, rating_band)
        ORDER BY month, rating_band, n_games DESC
    """)


def matchmaking_fairness(con, speed: str = "blitz"):
    """Win probability vs. rating gap (white_elo - black_elo), by rating band.

    A well-functioning matchmaker should rarely produce large gaps; this also
    reports how MANY games fall in each gap bucket, not just the win rate,
    since a win rate from 10 games is not the same claim as one from 100,000.
    """
    return con.execute(f"""
        SELECT
            {RATING_BANDS.replace('white_elo', 'LEAST(white_elo, black_elo)')} AS rating_band,
            {_bucket_expr('(white_elo - black_elo)', -400, 400, 16)} AS gap_bucket,
            round(avg(white_elo - black_elo), 0) AS avg_gap,
            count(*) AS n_games,
            round(avg(white_score), 4) AS white_win_rate
        FROM games
        WHERE speed = '{speed}' AND rated AND white_elo IS NOT NULL AND black_elo IS NOT NULL
        GROUP BY 1, 2
        HAVING count(*) >= 50
        ORDER BY 1, 2
    """)


def time_pressure(con, speed: str = "blitz"):
    """Loss rate vs. remaining clock time at move 1 (seconds), by rating band.

    Uses games with has_clock; the quality report's has_clock fraction tells
    us up front whether a given month has enough of these to be meaningful
    (clock annotations are sparse before ~2017).
    """
    return con.execute(f"""
        WITH exploded AS (
            SELECT {RATING_BANDS} AS rating_band, white_score,
                   unnest(clocks) AS clock_seconds,
                   generate_subscripts(clocks, 1) AS move_idx
            FROM games
            WHERE speed = '{speed}' AND rated AND has_clock
        )
        SELECT rating_band,
               {_bucket_expr('clock_seconds', 0, 180, 12)} AS clock_bucket,
               round(avg(clock_seconds), 0) AS avg_seconds_left,
               count(*) AS n_observations,
               round(avg(1 - white_score), 4) AS white_loss_rate
        FROM exploded
        WHERE move_idx <= 40
        GROUP BY 1, 2
        HAVING count(*) >= 200
        ORDER BY 1, 2
    """)


def clock_coverage_by_month(con):
    """% of rated games with clock data, by month -- sanity check before using time_pressure()."""
    return con.execute("""
        SELECT month, speed, count(*) AS n_games,
               round(avg(has_clock::int), 4) AS pct_has_clock,
               round(avg(has_eval::int), 4) AS pct_has_eval
        FROM games WHERE rated GROUP BY 1, 2 ORDER BY 1, 2
    """)


def _write(relation, out_dir: Path, name: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = relation.df()
    df.to_parquet(out_dir / f"{name}.parquet", index=False)
    print(f"  wrote {name}.parquet ({len(df):,} rows)")
    return len(df)


def build_all(con, out_dir: Path, speed: str = "blitz") -> dict:
    """Run every query and write dashboard-ready Parquet files into out_dir."""
    print(f"[analytics] building aggregates for speed={speed}")
    counts = {
        "opening_meta_by_month": _write(opening_meta_by_month(con, speed), out_dir, "opening_meta_by_month"),
        "matchmaking_fairness": _write(matchmaking_fairness(con, speed), out_dir, "matchmaking_fairness"),
        "clock_coverage_by_month": _write(clock_coverage_by_month(con), out_dir, "clock_coverage_by_month"),
    }
    tp_rows = _write(time_pressure(con, speed), out_dir, "time_pressure")
    counts["time_pressure"] = tp_rows
    if tp_rows == 0:
        print("  [note] time_pressure has 0 rows -- check clock_coverage_by_month.parquet; "
              "clock annotations are sparse before ~2017, try a later month")
    return counts
