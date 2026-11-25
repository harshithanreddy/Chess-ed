"""Feature engineering for the win/draw/loss model.

Leakage rule: a feature is allowed only if it is known BEFORE the game is
decided. played_at, elo, time control and rated/speed qualify. ECO/opening
also qualify in practice: Lichess assigns them once the position matches a
book line, typically within the first 4-10 plies, long before a 90-ply game
is decided. ply_count, termination, has_eval, has_clock and the move list
itself are NOT used: they are only fully known at game end and would leak
the outcome (e.g. short games correlate heavily with decisive results).
"""
from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "white_elo", "black_elo", "rating_diff", "avg_elo",
    "base_seconds", "increment_seconds", "rated", "eco",
]
LABEL_COLUMN = "outcome"  # 0 = black win, 1 = draw, 2 = white win
OUTCOME_NAMES = {0: "black_win", 1: "draw", 2: "white_win"}


def load_win_dataset(con, speed: str = "blitz", month: str | None = None,
                     min_games_per_eco: int = 500) -> pd.DataFrame:
    """One row per rated game with engineered features + outcome label.

    Rare ECOs are folded into 'OTHER' so the model (and a one-hot baseline)
    never has to extrapolate from a handful of examples of an opening.
    """
    where_month = f"AND month = '{month}'" if month else ""
    df = con.execute(f"""
        SELECT played_at, white_elo, black_elo, base_seconds, increment_seconds,
               rated, coalesce(eco, 'UNK') AS eco,
               CASE result WHEN '0-1' THEN 0 WHEN '1/2-1/2' THEN 1 WHEN '1-0' THEN 2 END AS outcome
        FROM games
        WHERE speed = '{speed}' AND rated
          AND white_elo IS NOT NULL AND black_elo IS NOT NULL
          AND base_seconds IS NOT NULL {where_month}
        ORDER BY played_at
    """).df()

    df["rating_diff"] = df["white_elo"] - df["black_elo"]
    df["avg_elo"] = (df["white_elo"] + df["black_elo"]) / 2
    eco_counts = df["eco"].value_counts()
    rare = eco_counts[eco_counts < min_games_per_eco].index
    df["eco"] = df["eco"].where(~df["eco"].isin(rare), "OTHER").astype("category")
    df["rated"] = df["rated"].astype(bool)
    return df


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Train on the earlier games, test on the later ones (no shuffling).

    A random split would leak information: Elo at time T is correlated with
    Elo at T+1 for the same player, so a random split lets the model see
    "future" ratings for players it already trained on. Splitting on time
    keeps the test set strictly after the training set, which is what a
    deployed model would actually face.
    """
    df = df.sort_values("played_at").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
