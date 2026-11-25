"""Needs duckdb, pandas (pip install -e .[dev]). LightGBM-dependent parts are
separate and skipped here; this file checks the SQL feature engineering only.
"""
import unittest

try:
    import duckdb
    import pandas as pd
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "duckdb/pandas not installed")
class RetentionFeatureTests(unittest.TestCase):
    def _games_df(self):
        import pandas as pd
        # player 'alice' as white: loses, loses, wins, loses, loses, loses -> max loss streak 3
        rows = [
            dict(month="2024-01", played_at="2024-01-01 00:00:00", white="alice", black="x1",
                 white_elo=1500, black_elo=1500, white_rating_diff=-5, black_rating_diff=5,
                 white_score=0.0, base_seconds=180, increment_seconds=0, rated=True, speed="blitz"),
            dict(month="2024-01", played_at="2024-01-02 00:00:00", white="alice", black="x2",
                 white_elo=1495, black_elo=1500, white_rating_diff=-5, black_rating_diff=5,
                 white_score=0.0, base_seconds=180, increment_seconds=0, rated=True, speed="blitz"),
            dict(month="2024-01", played_at="2024-01-03 00:00:00", white="alice", black="x3",
                 white_elo=1490, black_elo=1500, white_rating_diff=10, black_rating_diff=-10,
                 white_score=1.0, base_seconds=180, increment_seconds=0, rated=True, speed="blitz"),
            dict(month="2024-01", played_at="2024-01-04 00:00:00", white="alice", black="x4",
                 white_elo=1500, black_elo=1500, white_rating_diff=-5, black_rating_diff=5,
                 white_score=0.0, base_seconds=180, increment_seconds=0, rated=True, speed="blitz"),
            dict(month="2024-01", played_at="2024-01-05 00:00:00", white="alice", black="x5",
                 white_elo=1495, black_elo=1500, white_rating_diff=-5, black_rating_diff=5,
                 white_score=0.0, base_seconds=180, increment_seconds=0, rated=True, speed="blitz"),
            dict(month="2024-01", played_at="2024-01-06 00:00:00", white="alice", black="x6",
                 white_elo=1490, black_elo=1500, white_rating_diff=-5, black_rating_diff=5,
                 white_score=0.0, base_seconds=180, increment_seconds=0, rated=True, speed="blitz"),
            # bob plays in Jan only, does not return in Feb
            dict(month="2024-01", played_at="2024-01-10 00:00:00", white="bob", black="x7",
                 white_elo=1600, black_elo=1600, white_rating_diff=0, black_rating_diff=0,
                 white_score=0.5, base_seconds=180, increment_seconds=0, rated=True, speed="blitz"),
            # alice returns in Feb
            dict(month="2024-02", played_at="2024-02-01 00:00:00", white="alice", black="y1",
                 white_elo=1480, black_elo=1500, white_rating_diff=0, black_rating_diff=0,
                 white_score=0.5, base_seconds=180, increment_seconds=0, rated=True, speed="blitz"),
        ]
        return pd.DataFrame(rows).assign(played_at=lambda d: pd.to_datetime(d["played_at"]))

    def _con(self):
        con = duckdb.connect()
        games = self._games_df()
        con.register("games", games)
        con.execute("CREATE VIEW months AS SELECT DISTINCT month FROM games ORDER BY 1")
        return con

    def test_max_loss_streak_and_return_label(self):
        from metalens.ml.retention import build_player_month_features

        con = self._con()
        df = build_player_month_features(con, "2024-01", "2024-02", speed="blitz")
        df = df.set_index("player")
        self.assertEqual(df.loc["alice", "n_games"], 6)
        self.assertEqual(df.loc["alice", "max_loss_streak"], 3)  # losses 4,5,6 after the win on game 3
        self.assertEqual(df.loc["alice", "returned_next_month"], 1)
        self.assertEqual(df.loc["bob", "returned_next_month"], 0)

    def test_missing_label_month_returns_features_only(self):
        from metalens.ml.retention import build_player_month_features

        con = self._con()
        df = build_player_month_features(con, "2024-01", "2099-01", speed="blitz")
        self.assertNotIn("returned_next_month", df.columns)


if __name__ == "__main__":
    unittest.main()
