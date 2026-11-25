"""End-to-end: needs pyarrow + zstandard (pip install -e .[dev])."""
import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import zstandard
    HAVE = True
except ImportError:
    HAVE = False

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample.pgn"


@unittest.skipUnless(HAVE, "pyarrow/zstandard not installed")
class IngestE2E(unittest.TestCase):
    def test_ingest_is_correct_and_idempotent(self):
        from metalens.ingest import ingest_file

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "lichess_db_standard_rated_2016-01.pgn.zst"
            src.write_bytes(zstandard.ZstdCompressor().compress(SAMPLE.read_bytes()))
            out = tmp / "processed"
            m = ingest_file(src, workers=2, out_root=out)
            self.assertEqual(m["games_ok"], 4)
            table = pq.read_table(out / "month=2016-01" / "games")
            self.assertEqual(table.num_rows, 4)
            again = ingest_file(src, workers=2, out_root=out)  # skipped via _SUCCESS
            self.assertEqual(again["games_ok"], 4)


if __name__ == "__main__":
    unittest.main()
