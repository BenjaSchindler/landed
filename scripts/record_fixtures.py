"""Re-record all replay fixtures against the live OpenAI API.

Runs every eval case through the pipeline with LANDED_RECORD=1, overwriting the
hand-authored fixtures with real model outputs (same keys, same files). Run
this once with OPENAI_API_KEY set, then `python -m harness.eval` to grade
the live model.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["LANDED_RECORD"] = "1"

from harness.indexer import RepoIndex
from harness.llm import LLM
from harness.pipeline import Pipeline
from harness.schemas import FeedbackItem
from harness.server import _ensure_demo_repo

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    repo = _ensure_demo_repo()
    index = RepoIndex(repo, ROOT / "demo" / "releases.json")
    llm = LLM()
    if llm.mode != "live":
        sys.exit("No OpenAI credentials resolve — export OPENAI_API_KEY first.")
    pipeline = Pipeline(index, llm)

    cases = [json.loads(l) for l in (ROOT / "evals" / "cases.jsonl").read_text().splitlines() if l.strip()]
    for case in cases:
        item = FeedbackItem(id=case["id"], text=case["text"], channel=case.get("channel", "eval"))
        result = pipeline.analyze(item)
        print(f"  {case['id']:<10} recorded -> {result.verdict.verdict.value}")
    print("\nDone. Now run:  .venv/bin/python -m harness.eval")


if __name__ == "__main__":
    main()
