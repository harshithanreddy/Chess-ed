import unittest
from pathlib import Path

from metalens.parser import ParseError, parse_chunk, parse_game, split_games

SAMPLE = (Path(__file__).resolve().parents[1] / "data" / "sample" / "sample.pgn").read_text()


class ParserTests(unittest.TestCase):
    def test_sample_counts(self):
        recs, rej = parse_chunk(SAMPLE)
        self.assertEqual(len(recs), 4)
        self.assertEqual(rej, [])

    def test_fields_blitz_game(self):
        r = parse_chunk(SAMPLE)[0][0]
        self.assertEqual(r["game_id"], "abc12345")
        self.assertEqual(r["speed"], "blitz")
        self.assertEqual((r["white_elo"], r["black_elo"]), (1500, 1480))
        self.assertEqual(r["white_rating_diff"], 6)
        self.assertEqual(r["base_seconds"], 180)
        self.assertEqual(r["ply_count"], 12)
        self.assertTrue(r["moves"].startswith("e4 e5 Nf3 Nc6"))
        self.assertEqual(r["clocks"][:3], [180, 180, 178])
        self.assertEqual(r["white_score"], 1.0)

    def test_eval_flag_and_speed(self):
        r = parse_chunk(SAMPLE)[0][1]
        self.assertTrue(r["has_eval"])
        self.assertFalse(r["has_clock"])
        self.assertEqual(r["speed"], "rapid")
        self.assertEqual(r["increment_seconds"], 5)

    def test_bullet_tournament_and_draw(self):
        r = parse_chunk(SAMPLE)[0][2]
        self.assertEqual(r["speed"], "bullet")
        self.assertEqual(r["white_score"], 0.5)

    def test_anonymous_elo_is_none(self):
        r = parse_chunk(SAMPLE)[0][3]
        self.assertIsNone(r["white_elo"])
        self.assertEqual(r["black_elo"], 1750)
        self.assertEqual(r["moves"], "e4 e5 Nf3")

    def test_bad_result_is_quarantined(self):
        bad = SAMPLE.split("\n\n")[0].replace('[Result "1-0"]', '[Result "banana"]')
        recs, rej = parse_chunk(bad + "\n\n1. e4 e5 banana\n")
        self.assertEqual(recs, [])
        self.assertTrue(rej[0]["reason"].startswith("bad_result"))

    def test_missing_header_raises(self):
        with self.assertRaises(ParseError):
            parse_game('[Event "x"]\n\n1. e4 1-0\n')

    def test_split_games(self):
        self.assertEqual(len(split_games(SAMPLE)), 4)


if __name__ == "__main__":
    unittest.main()
