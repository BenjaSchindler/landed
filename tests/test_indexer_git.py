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


@unittest.skipUnless(shutil.which("git"), "git not available")
class TestStageOrderWarning(unittest.TestCase):
    """Listing the stages backwards makes pre-production look like what users
    run — the false "already fixed" again, from a config git cannot check.

    It can check direction, though: work reaches pre-production first, so a
    later stage whose newest commit predates production's is worth flagging.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = Path(self.tmp)
        git(self.repo, "init", "-q", "-b", "prod", ".")
        git(self.repo, "config", "user.email", "t@t.com")
        git(self.repo, "config", "user.name", "T")
        self.commit("shared", "2026-01-01T10:00:00+00:00")
        git(self.repo, "branch", "staging")
        # production drifts with old work staging never took, while staging
        # carries the newer commits — the shape a real repo has
        self.commit("old hotfix straight to prod", "2026-02-01T10:00:00+00:00")
        git(self.repo, "checkout", "-q", "staging")
        self.commit("recent work awaiting promotion", "2026-06-01T10:00:00+00:00")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def commit(self, message, when):
        (self.repo / "f.txt").write_text(message)
        git(self.repo, "add", ".")
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", message],
                       check=True, capture_output=True,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(self.repo),
                            "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when,
                            "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@t.com",
                            "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@t.com"})

    def test_correct_order_is_quiet_even_when_production_holds_extra_commits(self):
        # counts would misfire here: prod has a commit staging lacks
        index = RepoIndex(self.repo, None, ["prod", "staging"])
        self.assertIsNone(index.branch_warning)

    def test_inverted_order_is_flagged(self):
        index = RepoIndex(self.repo, None, ["staging", "prod"])
        self.assertIsNotNone(index.branch_warning)
        self.assertIn("prod", index.branch_warning)

    def test_a_single_stage_has_no_order_to_get_wrong(self):
        self.assertIsNone(RepoIndex(self.repo, None, ["staging"]).branch_warning)
