"""Eval runner: labeled feedback in, verdict quality out.

The headline metric is NOT accuracy — it's the false-"fixed" rate: how often
the tool tells support a problem is handled when it isn't. That mistake sends
a confident wrong answer to an end user, so the harness is tuned to abstain
(UNCERTAIN / NOT_FIXED) rather than over-claim, and this runner measures
exactly that asymmetry.

Usage:
  .venv/bin/python -m harness.eval            # live if credentials resolve, else replay
  .venv/bin/python -m harness.eval --limit 5
  LANDED_REPLAY=1 .venv/bin/python -m harness.eval   # force replay (fixture-covered cases only)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from .indexer import RepoIndex
from .llm import LLM
from .pipeline import Pipeline
from .schemas import FeedbackItem
from .server import _ensure_demo_repo

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "cases.jsonl"
FIXED_CLAIMS = {"already_fixed", "fix_coming"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    repo = _ensure_demo_repo()
    index = RepoIndex(repo, ROOT / "demo" / "releases.json")
    llm = LLM()
    pipeline = Pipeline(index, llm)
    print(f"mode: {llm.mode} | corpus: {len(index.commits)} commits\n")

    cases = [json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]
    if args.limit:
        cases = cases[: args.limit]

    rows, skipped = [], 0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    false_fixed, fixed_claims, bad_citations = 0, 0, 0

    for case in cases:
        item = FeedbackItem(id=case["id"], text=case["text"], channel=case.get("channel", "eval"))
        t0 = time.time()
        result = pipeline.analyze(item)
        elapsed = time.time() - t0

        if result.error and "fixture" in result.error:
            skipped += 1
            print(f"  {case['id']:<12} SKIP (no fixture — run live to cover)")
            continue

        got = result.verdict.verdict.value
        want = case["gold"]["verdict"]
        ok = got == want

        # extra checks on "fixed" claims: right version, valid citation
        version_ok = True
        if got in FIXED_CLAIMS:
            fixed_claims += 1
            if want not in FIXED_CLAIMS:
                false_fixed += 1
            gold_fixed_in = case["gold"].get("fixed_in")
            got_fixed_in = result.verdict.fix_commit.fixed_in if result.verdict.fix_commit else None
            if gold_fixed_in is not None and got_fixed_in != gold_fixed_in:
                version_ok = False
            if result.verdict.fix_commit and index.get_commit(result.verdict.fix_commit.sha) is None:
                bad_citations += 1

        per_class[want]["tp" if ok else "fn"] += 1
        if not ok:
            per_class[got]["fp"] += 1
        rows.append({"id": case["id"], "want": want, "got": got, "ok": ok,
                     "version_ok": version_ok, "seconds": round(elapsed, 1)})
        mark = "✓" if ok and version_ok else "✗"
        extra = "" if version_ok else " (wrong fix version)"
        print(f"  {case['id']:<12} {mark}  want={want:<14} got={got:<14} {elapsed:>5.1f}s{extra}")

    graded = [r for r in rows]
    if not graded:
        print("\nNo cases graded (all skipped). Set OPENAI_API_KEY and re-run.")
        return

    correct = sum(1 for r in graded if r["ok"] and r["version_ok"])
    print(f"\n== {correct}/{len(graded)} correct "
          f"({skipped} skipped) ==")
    print(f"false-'fixed' rate: {false_fixed}/{fixed_claims} fixed-claims wrong"
          + ("  ← the metric that matters" if fixed_claims else ""))
    print(f"invalid citations:  {bad_citations} (must be 0 — enforced by the harness)")
    print("\nper-class:")
    for cls, m in sorted(per_class.items()):
        prec = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 1.0
        rec = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else 1.0
        print(f"  {cls:<16} precision {prec:.2f}  recall {rec:.2f}  (n={m['tp'] + m['fn']})")

    out = ROOT / "evals" / "last_run.json"
    out.write_text(json.dumps({
        "mode": llm.mode, "graded": len(graded), "skipped": skipped,
        "correct": correct, "false_fixed": false_fixed, "fixed_claims": fixed_claims,
        "invalid_citations": bad_citations, "rows": rows,
    }, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
