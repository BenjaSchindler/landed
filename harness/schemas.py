"""Typed contracts between pipeline stages.

The two LLM-facing models (IntakeResult, Adjudication) are enforced with
structured outputs — the API validates against the JSON schema, so a parse
failure surfaces as an SDK error rather than a corrupt object downstream.
Everything else is internal state passed between deterministic stages.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- LLM outputs

class SearchTerms(BaseModel):
    user_terms: list[str] = Field(description="Search terms in the reporter's own vocabulary and language")
    dev_terms: list[str] = Field(description="English terms an engineer would use in a commit message for this symptom")


class IntakeResult(BaseModel):
    kind: Literal["bug", "feature_request", "question", "praise", "other"]
    summary: str = Field(description="One-line restatement of the report, in English")
    symptoms: list[str] = Field(description="Concrete observable behaviors reported, in English")
    app_version: Optional[str] = Field(description="App version normalized to digits-and-dots like 1.0.1, or null if not mentioned")
    platform: Literal["ios", "android", "web", "unknown"]
    language: str = Field(description="Language of the report as a two-letter code, e.g. es, en")
    specific_enough: bool = Field(description="False if the report is too vague to search for a matching change")
    missing_info: list[str] = Field(description="If not specific enough or version unknown: the exact questions to ask the reporter, in their language")
    search_terms: SearchTerms


class Adjudication(BaseModel):
    match_sha: Optional[str] = Field(description="Full sha of the ONE candidate that fixes the reported symptom, or null if none does")
    confidence: float = Field(description="0.0-1.0: how confident that the matched change fixes this exact symptom; 0.0 when match_sha is null")
    reasoning: str = Field(description="One or two plain-language sentences a support agent can read")
    symptom_addressed: Optional[str] = Field(description="Which reported symptom the change addresses, or null")


# ---------------------------------------------------------------- internal

class Verdict(str, Enum):
    ALREADY_FIXED = "already_fixed"          # fix is in a released version newer than the reporter's
    FIX_COMING = "fix_coming"                # fix merged; not yet in the reporter's hands (in review / next release)
    REGRESSION_SUSPECTED = "regression"      # reporter runs a version that should contain the fix
    NOT_FIXED = "not_fixed"                  # no matching change found
    UNCERTAIN = "uncertain"                  # weak match; a human should confirm
    NEEDS_INFO = "needs_info"                # report too vague to search
    NOT_A_BUG = "not_a_bug"                  # feature request / question / praise
    FAILED = "failed"                        # pipeline error; never guessed


class Commit(BaseModel):
    sha: str
    date: str               # ISO author date
    subject: str
    body: str
    files: list[str]
    fixed_in: Optional[str] = None   # first release tag containing this commit, e.g. "1.1.0"; None = unreleased


class Release(BaseModel):
    version: str
    status: Literal["released", "in_review", "internal"]
    released_at: Optional[str] = None
    store_eta: Optional[str] = None  # for in_review: estimated store availability


class Candidate(BaseModel):
    commit: Commit
    score: float


class VerdictResult(BaseModel):
    verdict: Verdict
    fix_commit: Optional[Commit] = None
    fix_release: Optional[Release] = None    # release containing the fix, if tagged
    confidence: float = 0.0
    reporter_version: Optional[str] = None
    version_unknown: bool = False            # verdict computed without knowing the reporter's version
    reasoning: str = ""


class FeedbackItem(BaseModel):
    id: str
    text: str
    channel: str = "in-app"      # where it came from; display only
    received_at: str = ""


class AnalysisResult(BaseModel):
    """Everything the UI and the eval runner need, in one object."""
    item: FeedbackItem
    intake: Optional[IntakeResult] = None
    candidates: list[Candidate] = []
    adjudication: Optional[Adjudication] = None
    verdict: VerdictResult
    reply_draft: Optional[str] = None
    reply_is_fallback: bool = False          # True when the templated fallback replaced a rejected LLM draft
    escalation_packet: Optional[str] = None  # markdown handoff for engineering
    error: Optional[str] = None
