"""Prompt builders. Each returns (system, user) so the LLM seam can key fixtures.

Design notes:
- Intake merges normalization + query expansion into one call: the expansion
  needs everything normalization extracts, and one call halves latency.
- Adjudication sees ONLY retrieved candidates and must cite one by sha or
  return null — the pipeline verifies the sha exists in the candidate set.
- The reply drafter receives a facts object and may not state anything beyond
  it; the pipeline regex-checks version strings against an allowlist after.
"""
from __future__ import annotations

import json

from .schemas import Candidate, IntakeResult, VerdictResult

INTAKE_SYSTEM = """You are the intake stage of a support tool for a mobile app team.
You read one piece of raw user feedback and produce a structured record.

Rules:
- kind: a complaint that something is wrong, broken, or got worse counts as "bug" even when vague ("the app is bad now"). Praise with no request is "praise"; how-do-I questions are "question".
- symptoms: concrete, observable behaviors only ("answers are lost when the app goes to background"), translated to English. Not paraphrases of feelings.
- app_version: normalize "v1.0.1", "version 1.0.1", "1.0.1 (build 47)" to "1.0.1". If the user only says "latest" or nothing, use null.
- specific_enough: false only when there is no concrete symptom to search for (e.g. "it crashes sometimes", "the app is bad"). A vague-but-concrete symptom ("stats screen is slow") IS specific enough.
- missing_info: written in the reporter's language, ready to send. Include a version question whenever app_version is null and kind is "bug".
- search_terms.dev_terms: the English words an engineer would put in a commit message that FIXES this symptom. Think about the mechanism, not just the surface: for "progress lost when switching apps" include terms like "persist", "background", "AppState", "save", "restore".
- Do not diagnose or guess causes. You only structure what was reported."""


CONTINUOUS_INTAKE_NOTE = """

This product is delivered continuously on the web: there are no app versions,
no builds and no store updates. Leave app_version null, and never put a
question about versions, updating, or reinstalling into missing_info — ask
about the browser, the page/URL, or what they were doing instead."""


def intake_prompt(text: str, channel: str, continuous: bool = False) -> tuple[str, str]:
    system = INTAKE_SYSTEM + (CONTINUOUS_INTAKE_NOTE if continuous else "")
    user = f"Channel: {channel}\nFeedback:\n<<<\n{text}\n>>>"
    return system, user


LOOKUP_SYSTEM = """You sort one message into two kinds of question.

A STATUS QUERY asks whether a specific problem has already been fixed — usually
written by a support agent checking before they answer someone. "Is the blank
PDF fixed?", "did we ship the login timeout fix?", "¿ya arreglaron el menú
mobile?". These are answerable from the change history.

Anything else is NOT a status query: how-do-I questions ("how can I export my
data?"), feature requests, complaints, praise, or a bare description of a
problem with no question about its status.

The distinction is whether they are asking about the STATE OF A FIX, not merely
whether the message mentions something broken.

subject: restate the problem as an observable symptom in English, the way a user
would experience it, so it can be matched against commit messages. If it is not
a status query, put an empty string."""


def lookup_prompt(text: str, summary: str) -> tuple[str, str]:
    user = f"Message:\n<<<\n{text}\n>>>\n\nNormalized as: {summary}"
    return LOOKUP_SYSTEM, user


ADJUDICATE_SYSTEM = """You decide whether one of the listed code changes fixes a user-reported symptom.

Hard rules:
- You may ONLY select from the numbered candidates below. match_sha must be the full sha copied exactly from a candidate, or null.
- A match means the change fixes the SPECIFIC reported symptom — not merely the same screen or feature area. "Fixes streak timezone math" does NOT match "streak counter overlaps the header".
- If several candidates plausibly match, pick the single best one.
- Prefer null over a stretch. A wrong "already fixed" answer costs the team a user's trust; a null just means a human looks at it.
- confidence always refers to the match — how sure you are that the selected change fixes this exact symptom. When match_sha is null, set confidence to 0.0.

Confidence calibration:
- 0.85-1.0: the commit message names this symptom or its direct mechanism.
- 0.6-0.85: the mechanism described clearly produces this symptom, but the symptom is not named.
- 0.4-0.6: related area, plausible but not clear. Usually better expressed as a lower value.
- below 0.4: same general area only. Use null instead unless something concrete links them.

reasoning: one or two sentences a non-technical support agent can read. Refer to the change by what it did in plain words, not by sha."""


def adjudicate_prompt(intake: IntakeResult, candidates: list[Candidate]) -> tuple[str, str]:
    lines = []
    for i, cand in enumerate(candidates, 1):
        c = cand.commit
        released = f"released in v{c.fixed_in}" if c.fixed_in else "not in any tagged release yet"
        lines.append(
            f"[{i}] sha: {c.sha}\n    date: {c.date[:10]}  ({released})\n"
            f"    subject: {c.subject}\n"
            + (f"    body: {c.body}\n" if c.body else "")
            + f"    files: {', '.join(c.files[:6])}"
        )
    user = (
        "User report (normalized):\n"
        f"- summary: {intake.summary}\n"
        f"- symptoms: {'; '.join(intake.symptoms)}\n"
        f"- platform: {intake.platform}, app version: {intake.app_version or 'unknown'}\n\n"
        "Candidate changes:\n\n" + "\n\n".join(lines)
    )
    return ADJUDICATE_SYSTEM, user


REPLY_SYSTEM = """You write the reply a support agent will send to an app user.

Hard rules:
- Write in the reporter's language (given as `language`).
- Every factual claim must come from the `facts` object. Do not invent version numbers, dates, features, or causes. Do not promise timelines beyond what facts contains.
- 50-110 words. Warm, direct, specific. No corporate filler ("we take your feedback seriously"), no excessive apology, at most one emoji and only if the user used them.
- If facts.ask_version is true, ask which app version they use (Settings > About).
- Sign off with: — {team_signature}"""

# Appended only for continuously delivered products. Kept out of REPLY_SYSTEM so
# the app-store prompt stays byte-identical: fixture keys hash the prompt, and
# editing the shared text silently invalidates every recorded reply — the
# keyless demo then falls back to templates and no eval notices, because the
# eval grades verdicts, not drafts.
CONTINUOUS_REPLY_NOTE = """

This product is a website that ships on every deploy: there is nothing to
install and no version to be on. Never mention app versions, updating,
reinstalling, or app stores.
- verdict already_fixed: the fix is live, so the action is to reload the page.
- verdict regression: the fix is live and STILL not working for this person.
  Never say it is resolved and never ask them to reload as if that settles it —
  the fix landed on facts.deployed_at and the symptom outlived it. Say it is
  being looked at with priority."""


def reply_prompt(item_text: str, intake: IntakeResult, verdict: VerdictResult,
                 team_signature: str, continuous: bool = False) -> tuple[str, str]:
    facts = {
        "verdict": verdict.verdict.value,
        "language": intake.language,
        "fix_summary": verdict.fix_commit.subject if verdict.fix_commit else None,
        "fixed_in_version": verdict.fix_release.version if verdict.fix_release else None,
        "fix_release_status": verdict.fix_release.status if verdict.fix_release else None,
        "store_eta": verdict.fix_release.store_eta if verdict.fix_release else None,
        "reporter_version": verdict.reporter_version,
        "ask_version": verdict.version_unknown,
        "questions_to_ask": intake.missing_info,
    }
    # appended, not interleaved: facts is serialized into the prompt, so adding a
    # key for app-store products would rekey and orphan every recorded fixture
    if continuous:
        facts["deployed_at"] = verdict.deployed_at
    system = REPLY_SYSTEM.replace("{team_signature}", team_signature)
    if continuous:
        system += CONTINUOUS_REPLY_NOTE
    user = (
        f"Original message from the user:\n<<<\n{item_text}\n>>>\n\n"
        f"facts:\n{json.dumps(facts, ensure_ascii=False, indent=1)}\n\n"
        "Write the reply."
    )
    return system, user
