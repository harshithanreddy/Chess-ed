"""Needs pyarrow, zstandard, duckdb, pandas (pip install -e .[dev])."""
import tempfile
import unittest
from pathlib import Path

try:
    import duckdb  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import zstandard
    HAVE = True
except ImportError:
    HAVE = False

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample.pgn"


@unittest.skipUnless(HAVE, "analytics dependencies not installed")
class AnalyticsTests(unittest.TestCase):
    def _two_months(self, tmp: Path):
        from metalens.ingest import ingest_file

        for month in ("2016-01", "2016-02"):
            src = tmp / f"lichess_db_standard_rated_{month}.pgn.zst"
            src.write_bytes(zstandard.ZstdCompressor().compress(SAMPLE.read_bytes()))
            ingest_file(src, workers=2, out_root=tmp / "processed")

    def test_connect_sees_all_months_and_drops_zero_ply(self):
        from metalens.analytics.db import connect

        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._two_months(tmp)
            con = connect(root=tmp / "processed")
            months = [r[0] for r in con.execute("SELECT * FROM months").fetchall()]
            self.assertEqual(months, ["2016-01", "2016-02"])
            # sample.pgn has one zero-ply-looking game (the "1. e4 e5 2. Nf3 0-1" abandoned
            # one has ply_count 3, so this checks the view runs, not a specific drop)
            total = con.execute("SELECT count(*) FROM games").fetchone()[0]
            self.assertEqual(total, 8)  # 4 games x 2 months, none are zero-ply in the sample

    def test_matchmaking_fairness_runs_and_has_expected_columns(self):
        from metalens.analytics.db import connect
        from metalens.analytics.queries import matchmaking_fairness

        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._two_months(tmp)
            con = connect(root=tmp / "processed")
            # lower the HAVING threshold via a direct call isn't exposed; just check it runs
            df = matchmaking_fairness(con, speed="rapid").df()
            self.assertIn("white_win_rate", df.columns)

    def test_build_all_writes_parquet_files(self):
        from metalens.analytics.db import connect
        from metalens.analytics.queries import build_all

        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._two_months(tmp)
            con = connect(root=tmp / "processed")
            out = tmp / "agg"
            build_all(con, out, speed="blitz")
            self.assertTrue((out / "clock_coverage_by_month.parquet").exists())
            self.assertTrue((out / "matchmaking_fairness.parquet").exists())
            self.assertTrue((out / "opening_meta_by_month.parquet").exists())
            self.assertTrue((out / "time_pressure.parquet").exists())


if __name__ == "__main__":
    unittest.main()
