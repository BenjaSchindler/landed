"""Build a searchable index of a repo's change history + release state.

Evidence corpus = git commits (subject, body, touched files) joined with a
release manifest (releases.json) that says which tagged versions actually
reached users. Retrieval is BM25 over a diacritics-stripped, code-aware
tokenization — at this corpus size (tens to low thousands of commits) lexical
search plus LLM query expansion beats maintaining an embedding index.
"""
from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi

from .schemas import Candidate, Commit, Release

RS = "\x1e"  # record separator between commits
US = "\x1f"  # unit separator between fields

_STOPWORDS = {
    # en
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "on", "at", "for", "and", "or", "it", "its", "this", "that", "with", "when",
    "my", "me", "i", "you", "your", "not", "no", "do", "does", "but", "so",
    # es
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en",
    "que", "se", "me", "mi", "te", "su", "es", "está", "esta", "por", "para",
    "con", "y", "o", "no", "al", "lo", "le", "cuando", "pero", "muy",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, strip diacritics, split camelCase/snake_case/paths."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # break camelCase before lowering so "AppState" -> "app state"
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = text.lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


class RepoIndex:
    def __init__(self, repo_path: str | Path, releases_path: str | Path | None = None,
                 branches: list[str] | None = None):
        """branches: deployment stages, most advanced first (e.g. ["main", "staging"]).

        Without it the corpus is whatever HEAD happens to point at, which makes
        the evidence depend on who last ran `git checkout`. Naming the stages
        also keeps work that never shipped — experimental branches — out of the
        corpus entirely, so it cannot be cited as a fix that reached anyone.
        """
        self.repo = Path(repo_path)
        self.branches = branches or []
        self.commits: list[Commit] = []
        self.releases: list[Release] = []
        self._by_sha: dict[str, Commit] = {}
        self._bm25: BM25Okapi | None = None
        self._load_releases(releases_path)
        self._load_commits()
        self._build_bm25()

    # -------------------------------------------------------------- loading

    def _load_releases(self, releases_path) -> None:
        if releases_path and Path(releases_path).exists():
            data = json.loads(Path(releases_path).read_text())
            self.releases = [Release(**r) for r in data["releases"]]
        else:
            # fallback: every git tag counts as a released version
            tags = [t for t in _git(self.repo, "tag", "--list").split() if t]
            self.releases = [Release(version=t.lstrip("v"), status="released") for t in tags]

    def _load_commits(self) -> None:
        fmt = f"{RS}%H{US}%aI{US}%s{US}%b{US}"
        # named stages define the corpus; otherwise it is HEAD's history
        refs = list(self.branches)
        raw = _git(self.repo, "log", *refs, "--no-merges", "--name-only", f"--pretty=format:{fmt}")
        # first tag (in semver order) containing each commit
        sha_to_tag: dict[str, str] = {}
        tags = [t for t in _git(self.repo, "tag", "--list", "--sort=version:refname").split() if t]
        for tag in tags:
            for sha in _git(self.repo, "rev-list", tag).split():
                sha_to_tag.setdefault(sha, tag.lstrip("v"))
        # furthest stage containing each commit: walk least-advanced first so the
        # more advanced branch overwrites, leaving "as far as this change got"
        sha_to_stage: dict[str, str] = {}
        for branch in reversed(refs):
            for sha in _git(self.repo, "rev-list", branch).split():
                sha_to_stage[sha] = branch

        for record in raw.split(RS):
            if not record.strip():
                continue
            sha, date, subject, body_and_files = record.split(US, 3)
            body, _, files_blob = body_and_files.partition(US)
            files = [f.strip() for f in files_blob.strip().splitlines() if f.strip()]
            self.commits.append(Commit(
                sha=sha, date=date, subject=subject.strip(), body=body.strip(),
                files=files, fixed_in=sha_to_tag.get(sha),
                stage=sha_to_stage.get(sha),
            ))
        self._by_sha = {c.sha: c for c in self.commits}

    def _build_bm25(self) -> None:
        docs = []
        for c in self.commits:
            path_terms = " ".join(p.replace("/", " ").replace(".", " ") for p in c.files)
            docs.append(tokenize(f"{c.subject} {c.body} {path_terms}"))
        self._bm25 = BM25Okapi(docs) if docs else None

    # -------------------------------------------------------------- queries

    def search(self, terms: list[str], top_k: int = 12) -> list[Candidate]:
        if not self._bm25:
            return []
        query = tokenize(" ".join(terms))
        if not query:
            return []
        scores = self._bm25.get_scores(query)
        ranked = sorted(zip(self.commits, scores), key=lambda t: t[1], reverse=True)
        return [Candidate(commit=c, score=round(float(s), 3)) for c, s in ranked[:top_k] if s > 0]

    def get_commit(self, sha: str) -> Commit | None:
        if sha in self._by_sha:
            return self._by_sha[sha]
        # tolerate short shas from the model
        matches = [c for s, c in self._by_sha.items() if s.startswith(sha)]
        return matches[0] if len(matches) == 1 else None

    def release_for(self, version: str | None) -> Release | None:
        if version is None:
            return None
        version = version.lstrip("v")
        for r in self.releases:
            if r.version == version:
                return r
        return None

    def latest_released(self) -> Release | None:
        released = [r for r in self.releases if r.status == "released"]
        return max(released, key=lambda r: semver_key(r.version)) if released else None


def semver_key(version: str) -> tuple[int, ...]:
    """'1.2.0' -> (1,2,0); tolerant of 'v' prefixes and missing parts."""
    parts = re.findall(r"\d+", version or "")
    return tuple(int(p) for p in parts[:3]) + (0,) * max(0, 3 - len(parts))
