"""Fast, dependency-free parser for Lichess PGN dumps.

It does NOT replay moves on a board (that is what makes python-chess slow); it
extracts headers, SAN move text, clocks and eval flags with regex/str ops.
Trade-off: no legality checking. Validity is covered by header/result checks
and by a cross-check test against python-chess on the sample.
"""
from __future__ import annotations

import re
from datetime import datetime

_HEADER = re.compile(r'^\[(\w+) "(.*)"\]\s*$')
_CLK = re.compile(r"\[%clk (\d+):(\d{2}):(\d{2})\]")
_COMMENT = re.compile(r"\{[^}]*\}")
_MOVENUM = re.compile(r"\d+\.(?:\.\.)?")
_GAME_START = re.compile(r"^(?=\[Event )", re.MULTILINE)

RESULT_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
_RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
_REQUIRED = ("Site", "White", "Black", "Result", "UTCDate", "UTCTime")
_SPEEDS = ("ultrabullet", "bullet", "blitz", "rapid", "classical", "correspondence")


class ParseError(ValueError):
    pass


def _int_or_none(v: str | None) -> int | None:
    if v is None:
        return None
    try:
        return int(v)  # handles "+6", "-6", "1500"
    except ValueError:
        return None  # "?" for anonymous / provisional


def _speed(event: str) -> str:
    e = event.lower()
    for s in _SPEEDS:
        if s in e:
            return s
    return "other"


def _time_control(tc: str | None) -> tuple[int | None, int | None]:
    if not tc or tc == "-":
        return None, None
    try:
        base, inc = tc.split("+")
        return int(base), int(inc)
    except ValueError:
        return None, None


def parse_game(text: str) -> dict:
    """Parse one game's text into a flat record. Raises ParseError if invalid."""
    headers: dict[str, str] = {}
    move_lines: list[str] = []
    for line in text.split("\n"):
        if line.startswith("["):
            m = _HEADER.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
        elif line.strip():
            move_lines.append(line)

    for key in _REQUIRED:
        if key not in headers:
            raise ParseError(f"missing_header:{key}")

    result = headers["Result"]
    if result not in RESULT_SCORE:
        raise ParseError(f"bad_result:{result[:20]}")

    try:
        d, t = headers["UTCDate"], headers["UTCTime"]
        played_at = datetime(
            int(d[0:4]), int(d[5:7]), int(d[8:10]), int(t[0:2]), int(t[3:5]), int(t[6:8])
        )
    except (ValueError, IndexError):
        raise ParseError("bad_datetime") from None

    movetext = " ".join(move_lines)
    clocks = [int(h) * 3600 + int(m) * 60 + int(s) for h, m, s in _CLK.findall(movetext)]
    has_eval = "[%eval" in movetext

    stripped = _MOVENUM.sub(" ", _COMMENT.sub(" ", movetext))
    moves = [tok for tok in stripped.split() if tok not in _RESULT_TOKENS and not tok.startswith("$")]

    event = headers.get("Event", "")
    base, inc = _time_control(headers.get("TimeControl"))
    return {
        "game_id": headers["Site"].rsplit("/", 1)[-1],
        "played_at": played_at,
        "speed": _speed(event),
        "rated": "rated" in event.lower() and "casual" not in event.lower(),
        "white": headers["White"],
        "black": headers["Black"],
        "white_elo": _int_or_none(headers.get("WhiteElo")),
        "black_elo": _int_or_none(headers.get("BlackElo")),
        "white_rating_diff": _int_or_none(headers.get("WhiteRatingDiff")),
        "black_rating_diff": _int_or_none(headers.get("BlackRatingDiff")),
        "result": result,
        "white_score": RESULT_SCORE[result],
        "eco": headers.get("ECO"),
        "opening": headers.get("Opening"),
        "base_seconds": base,
        "increment_seconds": inc,
        "termination": headers.get("Termination"),
        "ply_count": len(moves),
        "has_eval": has_eval,
        "has_clock": bool(clocks),
        "moves": " ".join(moves),
        "clocks": clocks or None,
    }


def split_games(text: str) -> list[str]:
    return [g for g in _GAME_START.split(text) if g.strip()]


def parse_chunk(text: str) -> tuple[list[dict], list[dict]]:
    """Parse many games. Returns (records, rejects). Never raises on bad games."""
    records, rejects = [], []
    for game in split_games(text):
        try:
            records.append(parse_game(game))
        except ParseError as e:
            rejects.append({"reason": str(e), "raw_head": game[:500]})
    return records, rejects
