"""Indexer tests that need a real repository.

Kept out of test_harness.py, which is hermetic by design: these build a
throwaway repo in a temp dir because what they pin — how the indexer reacts to
branch names — only exists in git's answers, not in our own logic.

Run: python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from harness.indexer import BranchNotFound, RepoIndex


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@unittest.skipUnless(shutil.which("git"), "git not available")
class TestBranchValidation(unittest.TestCase):
    """Pointing LANDED_REPO somewhere new with the old LANDED_BRANCHES still set
    is the ordinary way to get a bad stage name, and git's own error names
    neither the branch that is missing nor the ones that would work."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.repo = Path(cls.tmp)
        git(cls.repo, "init", "-q", "-b", "trunk", ".")
        git(cls.repo, "config", "user.email", "t@t.com")
        git(cls.repo, "config", "user.name", "T")
        (cls.repo / "f.txt").write_text("a")
        git(cls.repo, "add", ".")
        git(cls.repo, "commit", "-qm", "fix: the thing")
        git(cls.repo, "branch", "staging")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_a_missing_branch_is_named_not_buried_in_a_subprocess_error(self):
        with self.assertRaises(BranchNotFound) as caught:
            RepoIndex(self.repo, None, ["main", "staging"])
        missing, _, available = str(caught.exception).partition("available:")
        self.assertIn("main", missing)          # the one that is missing
        self.assertNotIn("staging", missing)    # and only that one
        self.assertIn("trunk", available)       # plus what they could have meant
        self.assertIn("staging", available)

    def test_branches_that_exist_are_indexed(self):
        index = RepoIndex(self.repo, None, ["trunk", "staging"])
        self.assertEqual(len(index.commits), 1)
        self.assertEqual(index.commits[0].stage, "trunk")

    def test_no_branches_named_falls_back_to_head(self):
        index = RepoIndex(self.repo, None)
        self.assertEqual(len(index.commits), 1)
        self.assertIsNone(index.commits[0].stage)
