"""The harness: feedback text in -> evidence-gated verdict + reply + handoff out.

Control flow (stages yield progress events for the UI):

  intake (LLM)  ->  route  ->  search (BM25)  ->  adjudicate (LLM)  ->  verdict (pure code)
                                                                          |
                                     reply draft (LLM, validated)  <------+-->  escalation packet (pure code)

Guardrails, in order of importance:
- The model never decides "fixed": it only links a symptom to a candidate
  commit. Whether that fix reached THIS user is version arithmetic, done in
  compute_verdict() with no model involved.
- A cited sha must exist in the candidate set (checked, one corrective retry).
- Reply drafts may not contain version numbers outside an allowlist; a
  violating draft is retried once, then replaced by a deterministic template.
- Any LLM failure degrades to FAILED/escalation — the tool never guesses.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterator

from .indexer import RepoIndex, semver_key
from .llm import LLM, LLMBadOutput, LLMUnavailable, ReplayMiss
from .prompts import adjudicate_prompt, intake_prompt, lookup_prompt, reply_prompt
from .schemas import (
    Adjudication, AnalysisResult, Candidate, FeedbackItem, IntakeResult,
    LookupTriage, Release, Verdict, VerdictResult,
)

CONF_FIXED = 0.70      # at or above: safe to tell the user it is fixed
CONF_MENTION = 0.40    # between: surface the candidate to a human, claim nothing
TOP_K = 12

# How the product reaches its users. "tags" is the app-store model this was
# built for: merged, then packaged, then reviewed, then installed. "continuous"
# is the web model, where merging to the branch IS shipping — there are no
# versions to compare, so the same question ("did the fix reach this person?")
# is answered with dates instead of semver.
DEPLOY_TAGS = "tags"
DEPLOY_CONTINUOUS = "continuous"

# On a continuous product a fix is live within minutes, but CDN caches, service
# workers and already-open tabs delay it, and report timestamps carry timezone
# slop. A report inside this window is treated as having crossed the deploy in
# flight rather than as evidence the fix failed. Reasoned, not measured — the
# same caveat DESIGN.md makes about the confidence thresholds.
DEPLOY_GRACE_DAYS = 2

# optional "v" prefix must be consumed: in "v1.1.0" there is no \b between "v"
# and "1", so a bare \d pattern would match the inner "1.0" and misread the
# version (found by the live eval run — a valid draft was wrongly rejected).
_VERSION_RE = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?)\b", re.IGNORECASE)


class Pipeline:
    def __init__(self, index: RepoIndex, llm: LLM,
                 team_signature: str = "The Ritmo team",
                 deploy_model: str = DEPLOY_TAGS):
        self.index = index
        self.llm = llm
        self.team_signature = team_signature
        self.deploy_model = deploy_model

    @property
    def continuous(self) -> bool:
        return self.deploy_model == DEPLOY_CONTINUOUS

    # ------------------------------------------------------------------ api

    def analyze(self, item: FeedbackItem) -> AnalysisResult:
        result = None
        for stage, payload in self.analyze_stages(item):
            if stage == "done":
                result = payload
        assert result is not None
        return result

    def analyze_stages(self, item: FeedbackItem) -> Iterator[tuple[str, object]]:
        """Yields (stage, payload) progress events; last event is ("done", AnalysisResult)."""
        try:
            yield from self._run(item)
        except (LLMUnavailable, LLMBadOutput, ReplayMiss) as e:
            result = AnalysisResult(
                item=item,
                verdict=VerdictResult(verdict=Verdict.FAILED,
                                      reasoning="Automatic analysis was unavailable. Nothing was guessed."),
                error=str(e),
            )
            result.escalation_packet = build_packet(result)
            # "pipeline_error", not "error": an SSE event named "error" triggers
            # EventSource.onerror on the client and kills the stream early.
            yield ("pipeline_error", {"message": friendly_error(e)})
            yield ("done", result)

    # ------------------------------------------------------------------ run

    def _run(self, item: FeedbackItem) -> Iterator[tuple[str, object]]:
        # 1. intake: normalize + expand
        system, user = intake_prompt(item.text, item.channel, self.continuous)
        intake = self.llm.structured("intake", system, user, IntakeResult, max_tokens=2048)
        yield ("intake", {"summary": intake.summary, "kind": intake.kind,
                          "language": intake.language, "app_version": intake.app_version})

        # 2. route non-bugs and unsearchable reports — no retrieval needed
        lookup = False
        if intake.kind != "bug":
            triage = self._lookup_triage(item, intake) if intake.kind == "question" else None
            if triage is None or not triage.is_status_query:
                verdict = VerdictResult(verdict=Verdict.NOT_A_BUG,
                                        reasoning=f"This is a {intake.kind.replace('_', ' ')}, not a bug report.")
                yield from self._finish(item, intake, [], None, verdict)
                return
            # the asker described a state, not a symptom; intake left symptoms
            # empty, so the restated subject is what retrieval has to work with
            lookup = True
            intake = intake.model_copy(update={"symptoms": [triage.subject]})
            yield ("lookup", {"subject": triage.subject})
        if not lookup and not intake.specific_enough:
            verdict = VerdictResult(
                verdict=Verdict.NEEDS_INFO,
                version_unknown=not self.continuous and intake.app_version is None,
                reasoning="Too vague to search the change history; ask the questions below first.")
            yield from self._finish(item, intake, [], None, verdict)
            return

        # 3. retrieve candidate changes
        terms = intake.search_terms.user_terms + intake.search_terms.dev_terms + intake.symptoms
        candidates = self.index.search(terms, TOP_K)
        yield ("search", {"count": len(candidates),
                          "corpus": len(self.index.commits),
                          "top": [c.commit.subject for c in candidates[:3]]})

        # 4. adjudicate (skip when nothing retrieved)
        adjudication = None
        if candidates:
            adjudication = self._adjudicate(intake, candidates)
            yield ("adjudicate", {"match": adjudication.match_sha is not None,
                                  "confidence": adjudication.confidence,
                                  "reasoning": adjudication.reasoning})

        # 5. verdict: pure arithmetic — semver, or deploy dates when continuous.
        # A pasted item carries no timestamp; treat it as having arrived now,
        # which is what a support agent pasting fresh feedback means.
        reported_at = item.received_at or datetime.now(timezone.utc).isoformat()
        verdict = compute_verdict(intake, adjudication, self.index,
                                  self.deploy_model, reported_at, lookup)
        yield ("verdict", {"verdict": verdict.verdict.value, "reasoning": verdict.reasoning})

        yield from self._finish(item, intake, candidates, adjudication, verdict, lookup)

    def _finish(self, item, intake, candidates, adjudication, verdict,
                lookup: bool = False) -> Iterator[tuple[str, object]]:
        result = AnalysisResult(item=item, intake=intake, candidates=candidates,
                                adjudication=adjudication, verdict=verdict, is_lookup=lookup)
        # a lookup has no reporter waiting on an answer — drafting one would
        # invent a recipient, and the verdict is already the whole answer
        if not lookup:
            result.reply_draft, result.reply_is_fallback = self._draft_reply(item, intake, verdict)
        if verdict.verdict in (Verdict.REGRESSION_SUSPECTED, Verdict.NOT_FIXED,
                               Verdict.UNCERTAIN, Verdict.FAILED):
            result.escalation_packet = build_packet(result)
        yield ("reply", {"fallback": result.reply_is_fallback, "skipped": lookup})
        yield ("done", result)

    def _lookup_triage(self, item: FeedbackItem, intake: IntakeResult):
        """Is this someone asking whether a fix shipped, or a user asking how to?

        Both arrive as kind="question" and are structurally identical — same
        empty symptoms, same populated search terms — so the difference has to
        be read semantically. Its own call, so the feedback-triage prompt stays
        byte-identical and keeps its recordings; any failure falls through to
        the old not_a_bug routing, which claims nothing.
        """
        system, user = lookup_prompt(item.text, intake.summary)
        try:
            return self.llm.structured("lookup_triage", system, user, LookupTriage,
                                       max_tokens=2048)
        except (LLMUnavailable, LLMBadOutput, ReplayMiss):
            return None

    # ------------------------------------------------------------ subroutines

    def _adjudicate(self, intake: IntakeResult, candidates: list[Candidate]) -> Adjudication:
        system, user = adjudicate_prompt(intake, candidates)
        adj = self.llm.structured("adjudicate", system, user, Adjudication,
                                  max_tokens=16000, effort="medium")
        if adj.match_sha is not None and not self._sha_in(adj.match_sha, candidates):
            # one corrective retry, then discard the invalid citation
            retry_user = user + ("\n\nYour previous answer cited a sha that is not in the "
                                 "candidate list. Answer again: copy a sha exactly from a "
                                 "candidate above, or use null.")
            adj = self.llm.structured("adjudicate_retry", system, retry_user, Adjudication,
                                      max_tokens=16000, effort="medium")
            if adj.match_sha is not None and not self._sha_in(adj.match_sha, candidates):
                adj = Adjudication(match_sha=None, confidence=0.0,
                                   reasoning="Automatic matching produced an invalid citation and was discarded.",
                                   symptom_addressed=None)
        return adj

    @staticmethod
    def _sha_in(sha: str, candidates: list[Candidate]) -> bool:
        return any(c.commit.sha == sha or c.commit.sha.startswith(sha) for c in candidates)

    def _draft_reply(self, item: FeedbackItem, intake: IntakeResult,
                     verdict: VerdictResult) -> tuple[str, bool]:
        allowed = version_allowlist(verdict)
        system, user = reply_prompt(item.text, intake, verdict, self.team_signature,
                                    self.continuous)
        try:
            draft = self.llm.text("reply", system, user, max_tokens=1024)
            if reply_versions_ok(draft, allowed):
                return draft, False
            retry_user = user + ("\n\nYour previous draft mentioned a version number not present "
                                 "in facts. Rewrite it using only versions from facts, or none.")
            draft = self.llm.text("reply_retry", system, retry_user, max_tokens=1024)
            if reply_versions_ok(draft, allowed):
                return draft, False
        except (LLMUnavailable, LLMBadOutput, ReplayMiss):
            pass
        return template_reply(intake, verdict, self.team_signature, self.deploy_model), True


# ---------------------------------------------------------------- pure logic

def compute_verdict(intake: IntakeResult, adjudication: Adjudication | None,
                    index: RepoIndex, deploy_model: str = DEPLOY_TAGS,
                    reported_at: str | None = None, lookup: bool = False) -> VerdictResult:
    """Where 'fixed' actually gets decided. No model calls — arithmetic only.

    Which arithmetic depends on how the product ships: semver against a release
    manifest for store-delivered apps, deploy dates for continuously deployed
    ones. The gating above it (no match, weak match) is shared.
    """
    continuous = deploy_model == DEPLOY_CONTINUOUS
    reporter_version = intake.app_version
    # a continuously deployed product has no version for the reporter to read
    # off a settings screen, and a lookup has no reporter at all — neither may
    # produce a verdict that asks someone which version they are on
    base = (dict(reporter_version=None, version_unknown=False) if (continuous or lookup) else
            dict(reporter_version=reporter_version, version_unknown=reporter_version is None))

    if adjudication is None or adjudication.match_sha is None or adjudication.confidence < CONF_MENTION:
        return VerdictResult(verdict=Verdict.NOT_FIXED, **base,
                             reasoning="No change in the repository history matches this symptom.")

    commit = index.get_commit(adjudication.match_sha)
    if commit is None:  # defensive: _adjudicate already validated
        return VerdictResult(verdict=Verdict.NOT_FIXED, **base,
                             reasoning="No verifiable matching change.")

    if adjudication.confidence < CONF_FIXED:
        return VerdictResult(
            verdict=Verdict.UNCERTAIN, fix_commit=commit, confidence=adjudication.confidence, **base,
            reasoning=f"A change from {commit.date[:10]} might be related ({adjudication.reasoning}) "
                      "— not confident enough to tell the user it is fixed.")

    conf = adjudication.confidence
    if continuous:
        return _continuous_verdict(commit, conf, base, reported_at, adjudication.reasoning,
                                   live_branch=(index.branches[0] if index.branches else None))

    if commit.fixed_in is None:
        return VerdictResult(
            verdict=Verdict.FIX_COMING, fix_commit=commit, confidence=conf, **base,
            reasoning=f"Fixed on {commit.date[:10]} ({adjudication.reasoning}) "
                      "but the fix is not in any tagged release yet — it ships with the next one.")

    release = index.release_for(commit.fixed_in) or Release(version=commit.fixed_in, status="internal")
    if release.status != "released":
        eta = f", expected in stores around {release.store_eta}" if release.store_eta else ""
        return VerdictResult(
            verdict=Verdict.FIX_COMING, fix_commit=commit, fix_release=release, confidence=conf, **base,
            reasoning=f"Fixed on {commit.date[:10]} and packaged in v{release.version}, "
                      f"which is not in users' hands yet ({release.status}{eta}).")

    # fix is in a released version
    if lookup:
        released = f", released {release.released_at}" if release.released_at else ""
        return VerdictResult(
            verdict=Verdict.ALREADY_FIXED, fix_commit=commit, fix_release=release,
            confidence=conf, **base,
            reasoning=f"Fixed on {commit.date[:10]} ({adjudication.reasoning}) and shipped in "
                      f"v{release.version}{released} — anyone on that version or newer has it.")
    if reporter_version is None:
        return VerdictResult(
            verdict=Verdict.ALREADY_FIXED, fix_commit=commit, fix_release=release,
            confidence=conf, reporter_version=None, version_unknown=True,
            reasoning=f"Fixed in v{release.version} (released). The reporter's version is unknown — "
                      "the reply asks for it and suggests updating.")
    if semver_key(reporter_version) < semver_key(release.version):
        return VerdictResult(
            verdict=Verdict.ALREADY_FIXED, fix_commit=commit, fix_release=release,
            confidence=conf, reporter_version=reporter_version, version_unknown=False,
            reasoning=f"Fixed in v{release.version}, and the reporter runs v{reporter_version} — "
                      "updating the app resolves this.")
    return VerdictResult(
        verdict=Verdict.REGRESSION_SUSPECTED, fix_commit=commit, fix_release=release,
        confidence=conf, reporter_version=reporter_version, version_unknown=False,
        reasoning=f"The reporter runs v{reporter_version}, which already contains the "
                  f"v{release.version} fix for this symptom — either the fix did not work "
                  "or the bug returned. Engineering should see this.")


def _continuous_verdict(commit, conf: float, base: dict, reported_at: str | None,
                        why: str, live_branch: str | None = None) -> VerdictResult:
    """Merged is shipped, so the question becomes *when*, not *which version*.

    A fix that went live before the reporter hit the symptom is the continuous
    analogue of a reporter running a version that already contains the fix:
    either it did not work or the bug came back. A fix that landed after they
    wrote in is simply already fixed.

    "Shipped" means reaching the branch that is actually deployed. When stages
    are named, a commit that only got as far as pre-production is merged but not
    in anyone's hands — the same fix_coming the tagged model gets from a release
    that is built but not yet in stores.
    """
    if live_branch and commit.stage and commit.stage != live_branch:
        return VerdictResult(
            verdict=Verdict.FIX_COMING, fix_commit=commit, confidence=conf, **base,
            reasoning=f"Fixed on {commit.date[:10]} ({why}), but the change is only on "
                      f"{commit.stage} — it has not been deployed to {live_branch} yet, "
                      "so it is not in users' hands.")

    deployed_at = commit.date[:10]
    deployed, reported = _parse_iso(commit.date), _parse_iso(reported_at)
    if deployed and reported and (reported - deployed).days > DEPLOY_GRACE_DAYS:
        return VerdictResult(
            verdict=Verdict.REGRESSION_SUSPECTED, fix_commit=commit, deployed_at=deployed_at,
            confidence=conf, **base,
            reasoning=f"The fix went live on {deployed_at} ({why}), before this report arrived "
                      f"on {reported_at[:10]} — either it did not work or the bug returned. "
                      "Engineering should see this.")
    return VerdictResult(
        verdict=Verdict.ALREADY_FIXED, fix_commit=commit, deployed_at=deployed_at,
        confidence=conf, **base,
        reasoning=f"Fixed and deployed on {deployed_at} ({why}). The change is already live, "
                  "so reloading picks it up — there is no update for the user to install.")


def _parse_iso(value: str | None):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def version_allowlist(verdict: VerdictResult) -> set[str]:
    allowed = set()
    if verdict.fix_release:
        allowed.add(verdict.fix_release.version)
    if verdict.fix_commit and verdict.fix_commit.fixed_in:
        allowed.add(verdict.fix_commit.fixed_in)
    if verdict.reporter_version:
        allowed.add(verdict.reporter_version)
    return allowed


def reply_versions_ok(draft: str, allowed: set[str]) -> bool:
    return all(v in allowed for v in _VERSION_RE.findall(draft))


def friendly_error(e: Exception) -> str:
    if isinstance(e, ReplayMiss):
        return "Demo mode has no recorded answer for this text. Run with an API key for live analysis."
    if isinstance(e, LLMUnavailable):
        return "The analysis service is unreachable right now. The item was kept — try again in a minute."
    return "Analysis could not finish. The item was kept and nothing was guessed."


# ------------------------------------------------------- deterministic output

_TEMPLATES: dict[tuple[str, str], str] = {
    ("already_fixed", "es"): "¡Gracias por avisarnos! Esto ya está corregido en la versión {fix_version}. {version_line} — {sig}",
    ("already_fixed", "en"): "Thanks for flagging this! It's already fixed in version {fix_version}. {version_line} — {sig}",
    ("fix_coming", "es"): "¡Gracias por avisarnos! Ya encontramos y corregimos este problema; el arreglo llega en la próxima actualización{eta}. — {sig}",
    ("fix_coming", "en"): "Thanks for reporting this! We found and fixed this issue; the fix arrives in the next update{eta}. — {sig}",
    ("regression", "es"): "Gracias por el aviso. Lo estamos revisando con prioridad y te contactaremos apenas tengamos novedades. — {sig}",
    ("regression", "en"): "Thanks for the heads-up. We're looking into this with priority and will get back to you as soon as we know more. — {sig}",
    ("not_fixed", "es"): "Gracias por contarnos. Pasamos tu reporte directamente al equipo para investigarlo y te avisaremos cuando haya novedades. — {sig}",
    ("not_fixed", "en"): "Thanks for letting us know. We've passed your report straight to the team to investigate and will follow up when we know more. — {sig}",
    ("uncertain", "es"): "Gracias por contarnos. Pasamos tu reporte directamente al equipo para investigarlo y te avisaremos cuando haya novedades. — {sig}",
    ("uncertain", "en"): "Thanks for letting us know. We've passed your report straight to the team to investigate and will follow up when we know more. — {sig}",
    ("needs_info", "es"): "¡Gracias por escribirnos! Para encontrar la causa necesitamos un poco más de detalle: {questions} — {sig}",
    ("needs_info", "en"): "Thanks for writing in! To track this down we need a little more detail: {questions} — {sig}",
    ("not_a_bug", "es"): "¡Gracias por la sugerencia! La anotamos para el equipo de producto. — {sig}",
    ("not_a_bug", "en"): "Thanks for the suggestion! We've noted it for the product team. — {sig}",
    # continuous delivery: nothing to install, nothing to ask a version for
    ("already_fixed/continuous", "es"): "¡Gracias por avisarnos! Esto ya está corregido y el arreglo está publicado desde el {deployed_at}. Recarga la página y debería quedar resuelto. — {sig}",
    ("already_fixed/continuous", "en"): "Thanks for flagging this! It's already fixed, and the change went live on {deployed_at}. Reload the page and you should be all set. — {sig}",
}


def template_reply(intake: IntakeResult, verdict: VerdictResult, sig: str,
                   deploy_model: str = DEPLOY_TAGS) -> str:
    lang = "es" if (intake.language or "").lower().startswith("es") else "en"
    key = verdict.verdict.value
    template = (_TEMPLATES.get((f"{key}/{deploy_model}", lang))
                or _TEMPLATES.get((key, lang)) or _TEMPLATES.get(("not_fixed", lang)))
    eta = ""
    if verdict.fix_release and verdict.fix_release.store_eta:
        eta = (f" (estimada para el {verdict.fix_release.store_eta})" if lang == "es"
               else f" (expected around {verdict.fix_release.store_eta})")
    if verdict.version_unknown:
        version_line = ("¿Nos cuentas qué versión usas (Ajustes > Acerca de)? Si es anterior, actualizar la app lo resuelve."
                        if lang == "es" else
                        "Could you tell us which version you're on (Settings > About)? If it's older, updating the app resolves it.")
    else:
        version_line = ("Actualiza la app desde la tienda y debería quedar resuelto."
                        if lang == "es" else "Update the app from the store and you should be all set.")
    return template.format(
        fix_version=verdict.fix_release.version if verdict.fix_release else "",
        version_line=version_line, eta=eta, sig=sig,
        deployed_at=verdict.deployed_at or "",
        questions=" ".join(intake.missing_info) if intake.missing_info else "",
    ).replace("  ", " ").strip()


def build_packet(result: AnalysisResult) -> str:
    """Engineering handoff, assembled deterministically from pipeline state."""
    item, intake, verdict = result.item, result.intake, result.verdict
    lines = [f"## Engineering handoff — feedback {item.id}", ""]
    if verdict.verdict == Verdict.REGRESSION_SUSPECTED and verdict.fix_release:
        lines += [f"**⚠ Suspected regression:** the v{verdict.fix_release.version} fix "
                  f"(\"{verdict.fix_commit.subject}\") should cover this, but the reporter on "
                  f"v{verdict.reporter_version} still sees the symptom.", ""]
    lines += [f"**Landed verdict:** {verdict.verdict.value}"
              + (f" (confidence {verdict.confidence:.2f})" if verdict.confidence else ""),
              f"**Why:** {verdict.reasoning}", ""]
    if intake:
        lines += [f"**Reporter:** {intake.platform}, app version {intake.app_version or 'unknown'}, "
                  f"language {intake.language}, via {item.channel}",
                  "**Symptoms:**"] + [f"- {s}" for s in intake.symptoms]
        if intake.missing_info:
            lines += ["", "**Missing info to request:**"] + [f"- {q}" for q in intake.missing_info]
        lines += ["", f"**Search terms tried:** {', '.join(intake.search_terms.dev_terms)}"]
    if result.candidates:
        lines += ["", "**Top candidate changes reviewed:**"]
        for cand in result.candidates[:3]:
            c = cand.commit
            flag = " ← flagged" if (verdict.fix_commit and c.sha == verdict.fix_commit.sha) else ""
            lines.append(f"- `{c.sha[:10]}` {c.date[:10]} — {c.subject}"
                         f" ({'v' + c.fixed_in if c.fixed_in else 'unreleased'}){flag}")
    if result.error:
        lines += ["", f"**Pipeline error:** {result.error}"]
    lines += ["", "**Original message:**", "> " + item.text.replace("\n", "\n> ")]
    return "\n".join(lines)
