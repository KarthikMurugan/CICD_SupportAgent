"""Happy-path tests for the standalone code review demo."""

from pathlib import Path
import sqlite3
import tempfile
import unittest

from support_tools import read_build_log, search_builds, summarize_builds, tail_build_log


class SupportToolsTests(unittest.TestCase):
    def test_reads_a_normal_build_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build.log").write_text("Build succeeded\n", encoding="utf-8")
            self.assertEqual(read_build_log(root, "build.log"), "Build succeeded\n")

    def test_searches_for_a_build(self):
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE builds (id INTEGER, name TEXT, status TEXT)")
            connection.execute("INSERT INTO builds VALUES (1, 'release', 'success')")
            self.assertEqual(search_builds(connection, "release"), [(1, "success")])

    def test_summarizes_two_builds(self):
        builds = [
            {"status": "success", "duration_seconds": 10},
            {"status": "failed", "duration_seconds": 30},
        ]
        self.assertEqual(summarize_builds(builds), {
            "success_rate": 0.5, "average_duration": 20.0,
        })

    def test_reads_last_two_lines(self):
        self.assertEqual(tail_build_log("one\ntwo\nthree", 2), ["two", "three"])


if __name__ == "__main__":
    unittest.main()
