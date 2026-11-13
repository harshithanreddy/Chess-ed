"""Guards the 'repo-relative paths only' rule: fails if machine-specific paths sneak in."""
import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
BAD = re.compile(r"[A-Za-z]:\\\\|[A-Za-z]:\\|/home/\w+|/Users/\w+|/mnt/")


class PathHygiene(unittest.TestCase):
    def test_no_machine_specific_paths(self):
        offenders = []
        for f in SRC.rglob("*.py"):
            for n, line in enumerate(f.read_text().splitlines(), 1):
                if BAD.search(line):
                    offenders.append(f"{f.name}:{n}: {line.strip()}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
