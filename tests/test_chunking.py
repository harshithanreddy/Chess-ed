import io
import unittest
from pathlib import Path

from metalens.ingest import infer_month, iter_chunks
from metalens.parser import split_games

SAMPLE = (Path(__file__).resolve().parents[1] / "data" / "sample" / "sample.pgn").read_bytes()


class ChunkTests(unittest.TestCase):
    def test_chunks_cut_on_game_boundaries_and_lose_nothing(self):
        data = SAMPLE.rstrip(b"\n") + b"\n\n"
        data = data * 50
        for size in (64, 500, 4096, 10**6):
            chunks = list(iter_chunks(io.BytesIO(data), chunk_bytes=size))
            self.assertEqual(b"".join(chunks), data)
            self.assertTrue(all(c.startswith(b"[Event ") for c in chunks))
            self.assertEqual(sum(len(split_games(c.decode())) for c in chunks), 200)

    def test_infer_month(self):
        self.assertEqual(infer_month(Path("lichess_db_standard_rated_2016-01.pgn.zst")), "2016-01")


if __name__ == "__main__":
    unittest.main()
