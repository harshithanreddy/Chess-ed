"""Needs duckdb, pandas, lightgbm, scikit-learn (pip install -e .[dev])."""
import unittest

try:
    import duckdb
    import lightgbm  # noqa: F401
    import sklearn  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "duckdb/lightgbm/scikit-learn not installed")
class WinModelSmokeTest(unittest.TestCase):
    def test_trains_on_synthetic_games_without_crashing(self):
        import numpy as np
        import pandas as pd

        from metalens.ml.win_model import train_and_evaluate

        rng = np.random.default_rng(0)
        n = 6000
        white_elo = rng.normal(1500, 300, n).clip(600, 2800)
        black_elo = rng.normal(1500, 300, n).clip(600, 2800)
        # outcome genuinely depends on rating_diff, so the model has real signal to learn
        p_white = 1 / (1 + 10 ** (-(white_elo - black_elo) / 400))
        u = rng.random(n)
        outcome = np.where(u < p_white * 0.95, 2, np.where(u < p_white * 0.95 + 0.07, 1, 0))
        games = pd.DataFrame({
            "played_at": pd.date_range("2024-01-01", periods=n, freq="min"),
            "white_elo": white_elo, "black_elo": black_elo,
            "base_seconds": 180, "increment_seconds": 0, "rated": True,
            "eco": rng.choice(["A00", "B01", "C41", "D00"], size=n),
            "result": np.where(outcome == 2, "1-0", np.where(outcome == 1, "1/2-1/2", "0-1")),
            "speed": "blitz", "month": "2024-01",
        })
        con = duckdb.connect()
        con.register("games", games)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            metrics = train_and_evaluate(con, speed="blitz", out_dir=Path(tmp))
            self.assertIn("lightgbm", metrics)
            self.assertLess(metrics["lightgbm"]["log_loss"], metrics["majority_class_baseline"]["log_loss"])
            self.assertTrue((Path(tmp) / "win_model.txt").exists())
            self.assertTrue((Path(tmp) / "win_model_calibration.parquet").exists())


if __name__ == "__main__":
    unittest.main()
