"""Needs pyarrow, zstandard, duckdb, pandas, pandera (pip install -e .[dev])."""
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import duckdb  # noqa: F401
    import pandera  # noqa: F401
    import pyarrow  # noqa: F401
    import zstandard
    HAVE = True
except ImportError:
    HAVE = False

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample.pgn"


@unittest.skipUnless(HAVE, "quality dependencies not installed")
class QualityTests(unittest.TestCase):
    def _ingest(self, tmp: Path):
        from metalens.ingest import ingest_file

        src = tmp / "lichess_db_standard_rated_2016-01.pgn.zst"
        src.write_bytes(zstandard.ZstdCompressor().compress(SAMPLE.read_bytes()))
        ingest_file(src, workers=2, out_root=tmp / "processed")

    def test_clean_month_passes(self):
        from metalens.quality import validate_month

        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._ingest(tmp)
            rep = validate_month("2016-01", root=tmp / "processed", agg_dir=tmp / "agg")
            self.assertEqual(rep["status"], "passed")
            self.assertEqual(rep["rows"], 4)
            self.assertTrue((tmp / "agg" / "quality" / "2016-01.json").exists())

    def test_duplicated_data_is_caught(self):
        from metalens.quality import QualityError, validate_month

        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._ingest(tmp)
            first = next((tmp / "processed" / "month=2016-01" / "games").rglob("*.parquet"))
            shutil.copy(first, first.with_name("duplicate.parquet"))
            with self.assertRaises(QualityError):
                validate_month("2016-01", root=tmp / "processed", agg_dir=tmp / "agg")


if __name__ == "__main__":
    unittest.main()
