"""Unit tests for the deterministic core of the harness.

Everything here runs hermetically — no git, no network, no fixtures. The
model is replaced by scripted stubs, because these tests pin exactly the
parts the model is never trusted with: version arithmetic, citation
validation, the reply version-allowlist, and the safe fallbacks.

Run: python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import unittest

from harness.indexer import semver_key, tokenize
from harness.llm import LLMUnavailable
from harness.pipeline import (
    DEPLOY_CONTINUOUS, Pipeline, compute_verdict, reply_versions_ok, template_reply,
    version_allowlist,
)
from harness.schemas import (
    Adjudication, Candidate, Commit, FeedbackItem, IntakeResult, Release,
    SearchTerms, Verdict, VerdictResult,
)

SHA_FIX = "a" * 40
SHA_OTHER = "b" * 40
SHA_BOGUS = "f" * 40


def make_commit(sha=SHA_FIX, fixed_in="1.1.0"):
    return Commit(sha=sha, date="2026-04-22T09:30:00-03:00",
                  subject="fix: persist in-progress answers when backgrounded",
                  body="", files=["src/exercise/runner.ts"], fixed_in=fixed_in)


def make_intake(version="1.0.1", kind="bug", specific=True, lang="en", missing=()):
    return IntakeResult(
        kind=kind, summary="answers lost when app backgrounded",
        symptoms=["answers lost when app goes to background"] if kind == "bug" else [],
        app_version=version, platform="android", language=lang,
        specific_enough=specific, missing_info=list(missing),
        search_terms=SearchTerms(user_terms=["lost answers"], dev_terms=["persist", "background"]),
    )


class FakeIndex:
    """Just enough of RepoIndex for compute_verdict and Pipeline."""

    def __init__(self, commits, releases):
        self.commits = commits
        self.releases = releases
        self._by_sha = {c.sha: c for c in commits}

    def search(self, terms, top_k=12):
        return [Candidate(commit=c, score=1.0) for c in self.commits[:top_k]]

    def get_commit(self, sha):
        if sha in self._by_sha:
            return self._by_sha[sha]
        matches = [c for s, c in self._by_sha.items() if s.startswith(sha)]
        return matches[0] if len(matches) == 1 else None

    def release_for(self, version):
        if version is None:
            return None
        version = version.lstrip("v")
        return next((r for r in self.releases if r.version == version), None)


RELEASED = Release(version="1.1.0", status="released", released_at="2026-05-12")
IN_REVIEW = Release(version="1.2.0", status="in_review", store_eta="2026-07-28")
INDEX = FakeIndex([make_commit()], [RELEASED, IN_REVIEW])


def adj(sha=SHA_FIX, conf=0.9):
    return Adjudication(match_sha=sha, confidence=conf,
                        reasoning="matches the reported symptom", symptom_addressed="answers lost")


class TestSemver(unittest.TestCase):
    def test_ordering_is_numeric_not_lexicographic(self):
        self.assertLess(semver_key("1.2.0"), semver_key("1.10.0"))
        self.assertLess(semver_key("1.9"), semver_key("1.10"))

    def test_tolerates_prefix_and_missing_parts(self):
        self.assertEqual(semver_key("v1.2"), (1, 2, 0))
        self.assertEqual(semver_key("1.2.0"), semver_key("v1.2.0"))


class TestTokenize(unittest.TestCase):
    def test_strips_diacritics_and_splits_camelcase(self):
        self.assertIn("temporizador", tokenize("el temporizador se reinició"))
        self.assertEqual(["app", "state"], tokenize("AppState"))

    def test_drops_stopwords_both_languages(self):
        self.assertNotIn("the", tokenize("the timer"))
        self.assertNotIn("cuando", tokenize("cuando vuelvo"))


class TestComputeVerdict(unittest.TestCase):
    """The claim itself — pure version arithmetic, no model in the loop."""

    def test_no_match_is_not_fixed(self):
        v = compute_verdict(make_intake(), None, INDEX)
        self.assertEqual(v.verdict, Verdict.NOT_FIXED)
        v = compute_verdict(make_intake(), adj(sha=None, conf=0.0), INDEX)
        self.assertEqual(v.verdict, Verdict.NOT_FIXED)

    def test_low_confidence_match_is_discarded(self):
        v = compute_verdict(make_intake(), adj(conf=0.2), INDEX)
        self.assertEqual(v.verdict, Verdict.NOT_FIXED)

    def test_mid_confidence_surfaces_candidate_without_claiming(self):
        v = compute_verdict(make_intake(), adj(conf=0.55), INDEX)
        self.assertEqual(v.verdict, Verdict.UNCERTAIN)
        self.assertIsNotNone(v.fix_commit)

    def test_untagged_fix_is_fix_coming(self):
        index = FakeIndex([make_commit(fixed_in=None)], [RELEASED])
        v = compute_verdict(make_intake(), adj(), index)
        self.assertEqual(v.verdict, Verdict.FIX_COMING)
        self.assertIsNone(v.fix_release)

    def test_fix_in_unreleased_version_is_fix_coming(self):
        index = FakeIndex([make_commit(fixed_in="1.2.0")], [RELEASED, IN_REVIEW])
        v = compute_verdict(make_intake(version="1.1.0"), adj(), index)
        self.assertEqual(v.verdict, Verdict.FIX_COMING)
        self.assertEqual(v.fix_release.version, "1.2.0")

    def test_fix_version_missing_from_manifest_is_not_claimed_released(self):
        index = FakeIndex([make_commit(fixed_in="9.9.9")], [RELEASED])
        v = compute_verdict(make_intake(), adj(), index)
        self.assertEqual(v.verdict, Verdict.FIX_COMING)

    def test_reporter_behind_released_fix_is_already_fixed(self):
        v = compute_verdict(make_intake(version="1.0.1"), adj(), INDEX)
        self.assertEqual(v.verdict, Verdict.ALREADY_FIXED)
        self.assertFalse(v.version_unknown)

    def test_reporter_on_fix_version_is_regression(self):
        v = compute_verdict(make_intake(version="1.1.0"), adj(), INDEX)
        self.assertEqual(v.verdict, Verdict.REGRESSION_SUSPECTED)

    def test_reporter_ahead_of_fix_is_regression(self):
        v = compute_verdict(make_intake(version="1.2.0"), adj(), INDEX)
        self.assertEqual(v.verdict, Verdict.REGRESSION_SUSPECTED)

    def test_unknown_reporter_version_still_claims_but_flags(self):
        v = compute_verdict(make_intake(version=None), adj(), INDEX)
        self.assertEqual(v.verdict, Verdict.ALREADY_FIXED)
        self.assertTrue(v.version_unknown)


class TestContinuousDelivery(unittest.TestCase):
    """Web products ship on merge: no tags, no reporter version, no store lag.

    The tagged model collapses every verdict here to fix_coming and tells the
    user a live fix "ships with the next release", so the claim is inverted
    exactly when it matters most.
    """

    # the demo commit is dated 2026-04-22; reports are placed relative to it
    BEFORE = "2026-04-20T10:00:00-03:00"   # reported before the fix landed
    JUST_AFTER = "2026-04-23T10:00:00-03:00"   # inside the propagation window
    LONG_AFTER = "2026-06-01T10:00:00-03:00"   # weeks after the fix went live

    def verdict(self, reported_at, fixed_in=None, conf=0.9):
        index = FakeIndex([make_commit(fixed_in=fixed_in)], [])
        return compute_verdict(make_intake(version=None), adj(conf=conf), index,
                               DEPLOY_CONTINUOUS, reported_at)

    def test_untagged_fix_is_live_not_pending(self):
        # the whole point: the tagged model calls this fix_coming
        v = self.verdict(self.BEFORE)
        self.assertEqual(v.verdict, Verdict.ALREADY_FIXED)
        self.assertEqual(v.deployed_at, "2026-04-22")
        self.assertIn("already live", v.reasoning)

    def test_symptom_seen_after_the_fix_went_live_is_a_regression(self):
        v = self.verdict(self.LONG_AFTER)
        self.assertEqual(v.verdict, Verdict.REGRESSION_SUSPECTED)
        self.assertEqual(v.deployed_at, "2026-04-22")

    def test_report_crossing_the_deploy_in_flight_is_not_a_regression(self):
        v = self.verdict(self.JUST_AFTER)
        self.assertEqual(v.verdict, Verdict.ALREADY_FIXED)

    def test_missing_report_date_never_invents_a_regression(self):
        for missing in (None, "", "not a date"):
            self.assertEqual(self.verdict(missing).verdict, Verdict.ALREADY_FIXED)

    def test_never_asks_a_web_user_for_an_app_version(self):
        for reported_at in (self.BEFORE, self.LONG_AFTER):
            v = self.verdict(reported_at)
            self.assertFalse(v.version_unknown)
            self.assertIsNone(v.reporter_version)
        # and the weak-match / no-match paths must not ask either
        self.assertFalse(self.verdict(self.BEFORE, conf=0.5).version_unknown)
        self.assertFalse(self.verdict(self.BEFORE, conf=0.1).version_unknown)

    def test_confidence_gating_still_applies(self):
        self.assertEqual(self.verdict(self.BEFORE, conf=0.5).verdict, Verdict.UNCERTAIN)
        self.assertEqual(self.verdict(self.BEFORE, conf=0.1).verdict, Verdict.NOT_FIXED)

    def test_reply_template_mentions_no_versions_and_no_store(self):
        v = self.verdict(self.BEFORE)
        for lang in ("es", "en"):
            text = template_reply(make_intake(version=None, lang=lang), v, "Team",
                                  DEPLOY_CONTINUOUS)
            self.assertTrue(reply_versions_ok(text, version_allowlist(v)), text)
            self.assertIn("2026-04-22", text)
            for banned in ("Settings > About", "Acerca de", "tienda", "store", "actualiza"):
                self.assertNotIn(banned.lower(), text.lower(), f"{banned!r} leaked: {text}")

    def test_continuous_support_leaves_the_app_store_prompt_byte_identical(self):
        # fixture keys hash the prompt, so perturbing the app-store prompt orphans
        # every recorded reply and the demo silently drops to templates — which no
        # verdict-graded eval notices. Continuous guidance may only append.
        from harness.prompts import reply_prompt
        verdict = VerdictResult(verdict=Verdict.ALREADY_FIXED, fix_commit=make_commit(),
                                fix_release=RELEASED, reporter_version="1.0.1")
        intake = make_intake()
        sys_tags, user_tags = reply_prompt("hi", intake, verdict, "Team")
        sys_cont, user_cont = reply_prompt("hi", intake, verdict, "Team", continuous=True)
        self.assertTrue(sys_cont.startswith(sys_tags), "continuous rewrote shared prompt text")
        self.assertNotEqual(sys_cont, sys_tags, "continuous guidance never reached the prompt")
        # facts is serialized into the user turn, so a new key rekeys it too
        self.assertNotIn("deployed_at", user_tags)
        self.assertIn("deployed_at", user_cont)

    def test_tagged_products_are_untouched(self):
        v = compute_verdict(make_intake(version="1.0.1"), adj(), INDEX)
        self.assertEqual(v.verdict, Verdict.ALREADY_FIXED)
        self.assertIsNone(v.deployed_at)
        self.assertEqual(v.reporter_version, "1.0.1")


class TestReplyGuardrail(unittest.TestCase):
    def test_allowlist_collects_all_known_versions(self):
        v = VerdictResult(verdict=Verdict.ALREADY_FIXED, fix_commit=make_commit(),
                          fix_release=RELEASED, reporter_version="1.0.1")
        self.assertEqual(version_allowlist(v), {"1.1.0", "1.0.1"})

    def test_draft_with_foreign_version_is_rejected(self):
        allowed = {"1.1.0"}
        self.assertTrue(reply_versions_ok("Fixed in version 1.1.0!", allowed))
        self.assertFalse(reply_versions_ok("Fixed in version 1.3.0!", allowed))
        self.assertTrue(reply_versions_ok("No versions mentioned.", allowed))

    def test_v_prefixed_version_is_read_whole_not_as_inner_fragment(self):
        # regression: "v1.1.0" used to be parsed as "1.0" (no \b between v and 1)
        # and a valid draft was rejected — caught by the first live eval run
        self.assertTrue(reply_versions_ok("Thanks for including v1.1.0.", {"1.1.0"}))
        self.assertFalse(reply_versions_ok("Fixed in v1.3.0.", {"1.1.0"}))

    def test_templates_pass_their_own_allowlist(self):
        for version, unknown in (("1.0.1", False), (None, True)):
            verdict = VerdictResult(verdict=Verdict.ALREADY_FIXED, fix_commit=make_commit(),
                                    fix_release=RELEASED, reporter_version=version,
                                    version_unknown=unknown)
            for lang in ("es", "en"):
                text = template_reply(make_intake(version=version, lang=lang), verdict, "Team")
                self.assertTrue(reply_versions_ok(text, version_allowlist(verdict)), text)

    def test_needs_info_template_includes_questions(self):
        verdict = VerdictResult(verdict=Verdict.NEEDS_INFO, version_unknown=True)
        text = template_reply(make_intake(version=None, missing=["Which version are you on?"]),
                              verdict, "Team")
        self.assertIn("Which version are you on?", text)


class TestSourceResolution(unittest.TestCase):
    """Which repo gets checked against which release manifest.

    Pointing at a foreign repo while silently keeping the demo's manifest made
    every real version unknown to release_for(), so no verdict could ever reach
    already_fixed or regression — the distinction the tool exists to make.
    """

    def setUp(self):
        from harness.server import resolve_sources
        self.resolve = resolve_sources
        self.demo_calls = []

    def demo(self):
        self.demo_calls.append(1)
        return "/demo/app-repo"

    def test_foreign_repo_without_manifest_falls_back_to_git_tags(self):
        repo, releases = self.resolve({"LANDED_REPO": "/my/repo"}, self.demo)
        self.assertEqual(repo, "/my/repo")
        self.assertIsNone(releases)  # None => RepoIndex treats every tag as released

    def test_foreign_repo_keeps_an_explicit_manifest(self):
        repo, releases = self.resolve(
            {"LANDED_REPO": "/my/repo", "LANDED_RELEASES": "/my/releases.json"}, self.demo)
        self.assertEqual((repo, releases), ("/my/repo", "/my/releases.json"))

    def test_demo_repo_still_gets_the_demo_manifest(self):
        repo, releases = self.resolve({}, self.demo)
        self.assertEqual(repo, "/demo/app-repo")
        self.assertTrue(releases.endswith("releases.json"))

    def test_demo_repo_is_not_generated_when_pointing_elsewhere(self):
        self.resolve({"LANDED_REPO": "/my/repo"}, self.demo)
        self.assertEqual(self.demo_calls, [])


class TestCredentialVerification(unittest.TestCase):
    """A key that merely exists must not be reported as a working "live" mode.

    _try_client() builds a client for any non-empty key, so a revoked one used
    to show a healthy green badge over an inbox where every item failed.
    """

    @staticmethod
    def _client(exc=None):
        class Models:
            def list(self):
                if exc:
                    raise exc
                return ["gpt-5.5"]

        class Client:
            models = Models()

        return Client()

    @staticmethod
    def _openai_error(cls):
        import httpx
        return cls("boom", response=httpx.Response(
            401, request=httpx.Request("GET", "http://x")), body=None)

    def test_working_key_reports_no_error(self):
        from harness.llm import LLM
        llm = LLM(mode="replay")          # constructed without touching the network
        llm.client = self._client()
        self.assertIsNone(llm._verify())

    def test_revoked_key_is_named_not_swallowed(self):
        import openai
        from harness.llm import LLM
        llm = LLM(mode="replay")
        llm.client = self._client(self._openai_error(openai.AuthenticationError))
        self.assertEqual(llm._verify(), "invalid API key")

    def test_transport_failure_is_reported_too(self):
        import openai
        from harness.llm import LLM
        llm = LLM(mode="replay")
        llm.client = self._client(openai.APIConnectionError(request=None))
        self.assertEqual(llm._verify(), "APIConnectionError")

    def test_replay_mode_never_probes(self):
        from harness.llm import LLM
        llm = LLM(mode="replay")
        self.assertIsNone(llm.credential_error)
        self.assertIsNone(llm.client)


class ScriptedLLM:
    """Returns queued outputs; raises for the reply so the fallback path is exercised
    deterministically unless text outputs are provided."""

    mode = "test"

    def __init__(self, structured=(), text=()):
        self.structured_queue = list(structured)
        self.text_queue = list(text)
        self.calls = []

    def structured(self, name, system, user, model_cls, **kw):
        self.calls.append(name)
        if not self.structured_queue:
            raise AssertionError(f"unexpected structured call {name}")
        return model_cls.model_validate(self.structured_queue.pop(0))

    def text(self, name, system, user, **kw):
        self.calls.append(name)
        if not self.text_queue:
            raise LLMUnavailable(f"{name}: scripted outage")
        return self.text_queue.pop(0)


def run_pipeline(llm, index=INDEX):
    item = FeedbackItem(id="t-1", text="answers gone after a call, on 1.0.1", channel="test")
    return Pipeline(index, llm).analyze(item)


class TestPipelineGuardrails(unittest.TestCase):
    def test_invalid_citation_gets_one_retry_then_succeeds(self):
        llm = ScriptedLLM(structured=[
            make_intake().model_dump(),
            adj(sha=SHA_BOGUS).model_dump(),   # cites something not in candidates
            adj(sha=SHA_FIX).model_dump(),     # corrected on retry
        ])
        result = run_pipeline(llm)
        self.assertIn("adjudicate_retry", llm.calls)
        self.assertEqual(result.verdict.verdict, Verdict.ALREADY_FIXED)

    def test_invalid_citation_twice_is_discarded_not_trusted(self):
        llm = ScriptedLLM(structured=[
            make_intake().model_dump(),
            adj(sha=SHA_BOGUS).model_dump(),
            adj(sha=SHA_BOGUS).model_dump(),
        ])
        result = run_pipeline(llm)
        self.assertEqual(result.verdict.verdict, Verdict.NOT_FIXED)
        self.assertIsNone(result.verdict.fix_commit)

    def test_reply_with_foreign_version_falls_back_to_template(self):
        llm = ScriptedLLM(
            structured=[make_intake().model_dump(), adj().model_dump()],
            text=["Update to 9.9.9 and it's fixed!", "Really, 9.9.9 fixes it."],
        )
        result = run_pipeline(llm)
        self.assertTrue(result.reply_is_fallback)
        self.assertTrue(reply_versions_ok(result.reply_draft,
                                          version_allowlist(result.verdict)))

    def test_llm_outage_mid_reply_still_produces_a_reply(self):
        llm = ScriptedLLM(structured=[make_intake().model_dump(), adj().model_dump()])
        result = run_pipeline(llm)
        self.assertEqual(result.verdict.verdict, Verdict.ALREADY_FIXED)
        self.assertTrue(result.reply_is_fallback)
        self.assertTrue(result.reply_draft)

    def test_intake_failure_degrades_to_failed_with_packet(self):
        class DeadLLM(ScriptedLLM):
            def structured(self, name, system, user, model_cls, **kw):
                raise LLMUnavailable("api down")

        result = run_pipeline(DeadLLM())
        self.assertEqual(result.verdict.verdict, Verdict.FAILED)
        self.assertIsNotNone(result.escalation_packet)

    def test_non_bug_skips_retrieval_entirely(self):
        llm = ScriptedLLM(structured=[make_intake(kind="praise").model_dump()])
        result = run_pipeline(llm)
        self.assertEqual(result.verdict.verdict, Verdict.NOT_A_BUG)
        self.assertNotIn("adjudicate", llm.calls)

    def test_vague_bug_asks_instead_of_searching(self):
        llm = ScriptedLLM(structured=[
            make_intake(specific=False, version=None,
                        missing=["What are you doing when it crashes?"]).model_dump()])
        result = run_pipeline(llm)
        self.assertEqual(result.verdict.verdict, Verdict.NEEDS_INFO)
        self.assertNotIn("adjudicate", llm.calls)


if __name__ == "__main__":
    unittest.main()
