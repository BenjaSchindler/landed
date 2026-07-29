"""Verify the harness logic by playing the model's role by hand.

For each labeled eval case this script supplies a realistic intake /
adjudication / reply, runs the REAL pipeline around those outputs, and asserts
the deterministic layers (routing, retrieval, sha validation, version
arithmetic, reply guardrails) produce the gold verdict. It isolates the
plumbing from the model: when a live eval number moves, this gate tells you
whether to look at the prompts or the pipeline.

By default nothing is written. With --write it also saves the hand-authored
outputs as replay fixtures — useful for bootstrapping before any live run.
The shipped fixtures are real recorded model outputs
(`LANDED_RECORD=1 python -m harness.eval` re-records them).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.indexer import RepoIndex
from harness.llm import write_authored_fixture
from harness.pipeline import Pipeline
from harness.schemas import FeedbackItem
from demo.make_demo_repo import main as make_repo

ROOT = Path(__file__).resolve().parent.parent


class AuthorLLM:
    """Duck-typed LLM seam: serves pre-written outputs and (optionally) records
    them as fixtures keyed by the exact prompts the pipeline builds."""

    def __init__(self, outputs: dict, write: bool = False):
        self.outputs = outputs
        self.write = write
        self.mode = "author"

    def _serve(self, name, system, user):
        if name not in self.outputs:
            raise AssertionError(f"pipeline requested unexpected call '{name}'")
        if self.write:
            write_authored_fixture(name, system, user, self.outputs[name])
        return self.outputs[name]

    def structured(self, name, system, user, model_cls, **kw):
        return model_cls.model_validate(self._serve(name, system, user))

    def text(self, name, system, user, **kw):
        return self._serve(name, system, user)


def intake(*, kind="bug", summary, symptoms, version=None, platform="unknown",
           lang="en", specific=True, missing=(), user_terms=(), dev_terms=()):
    return {
        "kind": kind, "summary": summary, "symptoms": list(symptoms),
        "app_version": version, "platform": platform, "language": lang,
        "specific_enough": specific, "missing_info": list(missing),
        "search_terms": {"user_terms": list(user_terms), "dev_terms": list(dev_terms)},
    }


def adjudication(index: RepoIndex, subject_marker: str | None, confidence: float,
                 reasoning: str, symptom: str | None = None):
    sha = None
    if subject_marker:
        matches = [c for c in index.commits if subject_marker in c.subject]
        assert len(matches) == 1, f"marker '{subject_marker}' matched {len(matches)} commits"
        sha = matches[0].sha
    return {"match_sha": sha, "confidence": confidence, "reasoning": reasoning,
            "symptom_addressed": symptom}


# ---------------------------------------------------------------- case data

ASK_VERSION_ES = "¿Qué versión de la app usas? La ves en Ajustes > Acerca de."
ASK_VERSION_EN = "Which app version are you on? You can see it in Settings > About."

PERSIST = "persist in-progress answers"
TIMER = "no longer restarts after switching"
NOTIF = "cancel scheduled reminders"
AVATAR = "crash when uploading avatar"
STATS = "stats page slow to open"
STREAK_TZ = "count days by local midnight"

R_PERSIST = "A change from April explicitly fixes answers being lost when the app goes to the background — the exact behavior reported."
R_TIMER = "A June change stops the breathing timer from restarting when the user switches apps, which is exactly this complaint."
R_NOTIF = "A change made reminders cancel before re-scheduling, fixing the duplicated daily notification described here."
R_AVATAR = "A March change fixes the crash when uploading a profile photo from the gallery on newer Android versions."
R_STATS = "A July change makes the stats page open fast with a year of history — the slow-open behavior reported."
R_STREAK_TZ = "A June change fixes streaks resetting when a habit is completed late at night, due to the day boundary being computed in UTC."
R_NONE_BATTERY = "No change in the history addresses battery drain or overnight background consumption."
R_NONE_LAYOUT = "The only streak-related fix changes timezone day-counting; nothing addresses the counter overlapping the header."


def build_cases(index: RepoIndex):
    """Per case: authored intake/adjudicate/reply keyed by eval case id."""
    A = lambda *args, **kw: adjudication(index, *args, **kw)  # noqa: E731
    return {
        "seed-001": {
            "intake": intake(summary="Exercise answers are erased when a phone call interrupts the exercise",
                             symptoms=["in-progress exercise answers are lost when the app goes to background during a phone call"],
                             version="1.0.1", platform="android", lang="en",
                             user_terms=["answers gone after phone call", "lost what I answered"],
                             dev_terms=["persist answers", "background", "AppState", "exercise runner", "restore state", "lost progress"]),
            "adjudicate": A(PERSIST, 0.93, R_PERSIST, "answers lost when app is backgrounded"),
            "reply": "Thanks for telling us — three times is three too many! 😤 Good news: this is already fixed in version 1.1.0. When a call comes in, your answers now save themselves automatically. Since you're on 1.0.1, just update Ritmo from the store and it shouldn't happen again. — The Ritmo team",
        },
        "seed-002": {
            "intake": intake(summary="Breathing timer restarts from zero when the user briefly switches apps",
                             symptoms=["breathing timer resets to zero after switching to another app and back"],
                             version="1.1.0", platform="ios", lang="en",
                             user_terms=["timer restarts", "check a text"],
                             dev_terms=["timer", "restart", "reset", "focus", "resume", "background", "breathing session"]),
            "adjudicate": A(TIMER, 0.92, R_TIMER, "timer restarts after switching apps"),
            "reply": "Thanks for flagging this — you're right, that's the opposite of calming. We found and fixed exactly this: the timer now picks up where it left off when you return. The fix is packaged in version 1.2.0, which is in store review and should reach you around 2026-07-28. No action needed on your side. — The Ritmo team",
        },
        "seed-003": {
            "intake": intake(summary="Beta tester on 1.2.0 still loses exercise answers when switching apps",
                             symptoms=["in-progress exercise answers are lost when the app goes to background"],
                             version="1.2.0", platform="unknown", lang="en",
                             user_terms=["losing my answers", "switch apps"],
                             dev_terms=["persist answers", "background", "exercise runner", "restore state"]),
            "adjudicate": A(PERSIST, 0.9, R_PERSIST, "answers lost when app is backgrounded"),
            "reply": "Thanks for flagging this — you're right, this shouldn't be happening on 1.2.0, so we're looking into it with priority. Your report is already with the team and we'll get back to you as soon as we know more. — The Ritmo team",
        },
        "seed-004": {
            "intake": intake(summary="Phone runs warm and battery drains overnight, app shown as top consumer",
                             symptoms=["high battery drain overnight attributed to the app", "device runs warm"],
                             version="1.1.0", platform="android", lang="en",
                             user_terms=["battery drains overnight", "phone warm"],
                             dev_terms=["battery", "drain", "background task", "wakelock", "power", "overnight"]),
            "adjudicate": A(None, 0.0, R_NONE_BATTERY),
            "reply": "Thanks for the detailed report — knowing it shows as the top consumer in battery settings helps a lot. This isn't something we've fixed yet, so we've sent your report straight to the team to investigate. We'll follow up as soon as we know more. — The Ritmo team",
        },
        "seed-005": {
            "intake": intake(kind="feature_request",
                             summary="Request to export history as a PDF to share with a therapist",
                             symptoms=[], lang="es", platform="unknown",
                             user_terms=["exportar historial PDF"], dev_terms=["export", "pdf", "history"]),
            "reply": "¡Qué buena idea, gracias por proponerla! 🙏 Poder llevar tu historial a tu psicóloga es justo el tipo de uso que queremos apoyar. La anotamos para el equipo de producto — no podemos prometer fecha, pero quedó registrada. Gracias por estos meses usando Ritmo. — The Ritmo team",
        },
        "seed-006": {
            "intake": intake(summary="The same daily reminder notification arrives twice each morning",
                             symptoms=["duplicate daily reminder notifications, about a minute apart"],
                             version=None, platform="unknown", lang="en",
                             missing=[ASK_VERSION_EN],
                             user_terms=["same reminder twice"],
                             dev_terms=["notification", "duplicate", "reminder", "schedule", "cancel"]),
            "adjudicate": A(NOTIF, 0.9, R_NOTIF, "duplicate reminder notifications"),
            "reply": "Thanks for telling us before turning notifications off! This exact bug — the reminder being scheduled twice — was fixed in version 1.1.0. Could you check which version you're on (Settings > About)? If it's older, updating from the store should stop the duplicates right away. — The Ritmo team",
        },
        "seed-007": {
            "intake": intake(summary="App crashes when uploading a profile photo from the gallery",
                             symptoms=["app closes immediately when picking a profile photo from the gallery"],
                             version="1.0.0", platform="android", lang="es",
                             user_terms=["se cae al subir foto"],
                             dev_terms=["crash", "avatar", "upload", "photo", "gallery", "permission"]),
            "adjudicate": A(AVATAR, 0.94, R_AVATAR, "crash when uploading profile photo"),
            "reply": "¡Hola! Gracias por avisarnos. Ese error al elegir la foto de la galería ya está corregido en la versión 1.0.1 — el problema era un permiso de Android. Como tienes la 1.0.0, actualiza Ritmo desde la Play Store y deberías poder subir tu foto sin problemas. — The Ritmo team",
        },
        "seed-008": {
            "intake": intake(summary="Streak counter overlaps the header text on a small screen",
                             symptoms=["streak counter visually overlaps the header text on iPhone SE"],
                             version="1.1.0", platform="ios", lang="en",
                             user_terms=["streak counter overlaps header"],
                             dev_terms=["streak", "layout", "overlap", "header", "small screen", "font"]),
            "adjudicate": A(None, 0.0, R_NONE_LAYOUT),
            "reply": "Thanks for catching that — small-screen layouts are easy to miss and this one slipped through. It isn't fixed yet, so we've filed it with the team with your device details attached. We'll let you know when a fix ships. — The Ritmo team",
        },
        "seed-009": {
            "intake": intake(summary="Stats screen takes several seconds of blank screen to open with a year of data",
                             symptoms=["stats screen takes 4-5 seconds to open for accounts with a lot of history"],
                             version="1.1.0", platform="ios", lang="en",
                             user_terms=["stats screen takes forever"],
                             dev_terms=["stats", "slow", "performance", "charts", "memoize", "history"]),
            "adjudicate": A(STATS, 0.88, R_STATS, "stats screen slow to open with large history"),
            "reply": "Thanks for reporting this — with a year of entries you hit exactly the case we just fixed: the stats screen was recomputing everything each time it opened. The fix is already done and ships with the next update. Nothing to do on your side; your data is untouched. — The Ritmo team",
        },
        "var-010": {
            "intake": intake(summary="Typed exercise answers are lost when switching briefly to another app",
                             symptoms=["in-progress exercise answers are lost when the app goes to background"],
                             version="1.0.0", platform="android", lang="en",
                             user_terms=["lose everything I typed"],
                             dev_terms=["persist answers", "background", "exercise runner", "restore state"]),
            "adjudicate": A(PERSIST, 0.92, R_PERSIST, "answers lost when app is backgrounded"),
            "reply": "Thanks for reporting this — losing what you wrote is the worst. Good news: this was fixed in version 1.1.0, which now saves your answers the moment you switch away. You're on 1.0.0, so updating the app from the store resolves it. — The Ritmo team",
        },
        "var-011": {
            "intake": intake(summary="Exercise answers are erased after switching apps; version unknown",
                             symptoms=["in-progress exercise answers are lost when the app goes to background"],
                             version=None, platform="unknown", lang="es",
                             missing=[ASK_VERSION_ES],
                             user_terms=["se borran las respuestas"],
                             dev_terms=["persist answers", "background", "exercise runner"]),
            "adjudicate": A(PERSIST, 0.9, R_PERSIST, "answers lost when app is backgrounded"),
            "reply": "¡Gracias por avisarnos! Esto ya está corregido en la versión 1.1.0. ¿Nos cuentas qué versión tienes? La ves en Ajustes > Acerca de. Si es anterior a la 1.1.0, actualizar la app desde la tienda lo deja resuelto. — The Ritmo team",
        },
        "var-012": {
            "intake": intake(summary="Breathing timer restarts by itself after answering a message",
                             symptoms=["breathing timer resets when switching to another app and back"],
                             version="1.0.1", platform="unknown", lang="es",
                             user_terms=["temporizador se reinicia"],
                             dev_terms=["timer", "restart", "reset", "focus", "resume", "breathing"]),
            "adjudicate": A(TIMER, 0.91, R_TIMER, "timer restarts after switching apps"),
            "reply": "¡Gracias por contarnos! Ya encontramos y corregimos este problema del temporizador. El arreglo viene en la versión 1.2.0, que está en revisión de la tienda y debería llegarte alrededor del 2026-07-28. No necesitas hacer nada. — The Ritmo team",
        },
        "var-013": {
            "intake": intake(summary="Duplicate daily reminder notifications every morning",
                             symptoms=["the same daily reminder notification arrives twice, a minute apart"],
                             version="1.0.1", platform="android", lang="en",
                             user_terms=["two identical reminders"],
                             dev_terms=["notification", "duplicate", "reminder", "cancel", "schedule"]),
            "adjudicate": A(NOTIF, 0.9, R_NOTIF, "duplicate reminder notifications"),
            "reply": "Thanks for the heads-up! That double reminder is already fixed in version 1.1.0 — the reminder was being scheduled twice. Since you're on 1.0.1, update Ritmo from the store and the duplicate notifications will stop. — The Ritmo team",
        },
        "var-014": {
            "intake": intake(summary="App crashes when uploading a profile picture from the gallery",
                             symptoms=["app crashes when uploading a profile picture from the gallery"],
                             version="1.0.1", platform="android", lang="en",
                             user_terms=["crashes upload profile picture"],
                             dev_terms=["crash", "avatar", "upload", "gallery", "permission", "profile photo"]),
            "adjudicate": A(AVATAR, 0.93, R_AVATAR, "crash when uploading profile photo"),
            "reply": "Thanks for the report — this shouldn't be happening on your version, so we've escalated it to the team with priority. We'll get back to you as soon as we know more. — The Ritmo team",
        },
        "var-015": {
            "intake": intake(summary="A 40-day streak reset to zero after completing the habit late at night",
                             symptoms=["streak resets to zero when a habit is completed late at night"],
                             version="1.1.0", platform="unknown", lang="es",
                             user_terms=["perdí mi racha", "completé en la noche"],
                             dev_terms=["streak", "reset", "midnight", "timezone", "day boundary", "UTC"]),
            "adjudicate": A(STREAK_TZ, 0.89, R_STREAK_TZ, "streak lost when completing late at night"),
            "reply": "¡Qué pena lo de tu racha de 40 días — y gracias por contarnos! 😭 Encontramos la causa: al completar un hábito tarde en la noche, el día se calculaba mal. Ya está corregido y el arreglo llega en la versión 1.2.0, que está en revisión de la tienda (aprox. 2026-07-28). — The Ritmo team",
        },
        "var-016": {
            "intake": intake(summary="Streak number overlaps the title text on a small screen",
                             symptoms=["streak counter overlaps the header text on iPhone SE"],
                             version="1.1.0", platform="ios", lang="es",
                             user_terms=["número se superpone con el título"],
                             dev_terms=["streak", "layout", "overlap", "header", "small screen"]),
            "adjudicate": A(None, 0.0, R_NONE_LAYOUT),
            "reply": "¡Gracias por el detalle! Tienes razón y aún no está corregido, así que pasamos tu reporte al equipo con los datos de tu dispositivo. Te avisamos cuando salga el arreglo. — The Ritmo team",
        },
        "var-017": {
            "intake": intake(summary="Stats screen takes several seconds to open for a long-time user",
                             symptoms=["stats screen very slow to open for accounts with much history"],
                             version="1.1.0", platform="unknown", lang="en",
                             user_terms=["stats take long to open"],
                             dev_terms=["stats", "slow", "performance", "charts", "memoize"]),
            "adjudicate": A(STATS, 0.87, R_STATS, "stats screen slow with large history"),
            "reply": "Thanks for letting us know! With a year and a half of data you hit exactly what we just fixed: the stats were recomputing everything on open. The fix is done and ships with the next update — nothing you need to do. — The Ritmo team",
        },
        "var-018": {
            "intake": intake(summary="Unspecified crashes", symptoms=[], specific=False,
                             missing=["What are you doing in the app when it crashes?",
                                      ASK_VERSION_EN],
                             user_terms=["crashes"], dev_terms=["crash"]),
            "reply": "Thanks for writing in — we'd love to track this down. Two quick questions: what are you doing in the app when it crashes, and which version are you on (Settings > About)? With that we can usually find the cause fast. — The Ritmo team",
        },
        "var-019": {
            "intake": intake(kind="praise", summary="Positive review after six months of use",
                             symptoms=[], lang="es", user_terms=[], dev_terms=[]),
            "reply": "¡Mil gracias! Mensajes así son los que le dan sentido a esto. Seis meses de constancia es mérito tuyo — nosotros solo ponemos la app 💙 — The Ritmo team",
        },
        "var-020": {
            "intake": intake(kind="question", summary="How to export or back up personal data",
                             symptoms=[], user_terms=["export data backup"], dev_terms=["export", "backup"]),
            # a how-to, not someone checking whether a fix shipped
            "lookup_triage": {"is_status_query": False, "subject": ""},
            "reply": "Thanks for asking! Right now Ritmo can share a weekly summary image, but there's no full data export yet — it's on our list and requests like yours push it up. We've noted yours. — The Ritmo team",
        },
        "var-021": {
            "intake": intake(summary="Phone hot and battery drained since leaving the app open overnight",
                             symptoms=["high battery drain and device heat when the app stays open overnight"],
                             version="1.1.0", platform="android", lang="en",
                             user_terms=["phone hot battery gone"],
                             dev_terms=["battery", "drain", "overnight", "wakelock", "background"]),
            "adjudicate": A(None, 0.0, R_NONE_BATTERY),
            "reply": "That sounds genuinely frustrating — thanks for telling us instead of just uninstalling. This isn't something we've fixed yet, so your report went straight to the team to investigate, with your device details attached. We'll follow up as soon as we know more. — The Ritmo team",
        },
        "var-022": {
            "intake": intake(summary="Breathing timer starts over when an alarm interrupts the session",
                             symptoms=["breathing timer restarts from the beginning after an interruption by another app"],
                             version="1.1.0", platform="unknown", lang="en",
                             user_terms=["timer starts over alarm"],
                             dev_terms=["timer", "restart", "reset", "resume", "interruption", "breathing"]),
            "adjudicate": A(TIMER, 0.88, R_TIMER, "timer restarts after interruption"),
            "reply": "Thanks for reporting this! We found and fixed exactly this: the timer now resumes where it left off after an interruption. The fix ships in version 1.2.0, currently in store review — it should reach you around 2026-07-28. — The Ritmo team",
        },
        "var-023": {
            "intake": intake(summary="Written reflections are erased when a phone call arrives; on 1.2.0",
                             symptoms=["in-progress exercise answers are lost when the app goes to background during a call"],
                             version="1.2.0", platform="unknown", lang="es",
                             user_terms=["se borró lo que escribí llamada"],
                             dev_terms=["persist answers", "background", "exercise runner", "restore"]),
            "adjudicate": A(PERSIST, 0.9, R_PERSIST, "answers lost when app is backgrounded"),
            "reply": "Qué rabia, tienes toda la razón — y en tu versión esto ya no debería pasar, así que lo estamos revisando con prioridad. Tu reporte ya está con el equipo y te contactamos apenas tengamos novedades. — The Ritmo team",
        },
        "var-024": {
            "intake": intake(kind="feature_request", summary="Request for a tablet/iPad layout",
                             symptoms=[], lang="en", user_terms=["tablet layout"], dev_terms=["tablet", "ipad", "layout"]),
            "reply": "Thanks for the suggestion! A proper tablet layout is noted for the product team — you're right that the iPad view isn't ideal yet. — The Ritmo team",
        },
        "var-025": {
            "intake": intake(summary="General complaint that the app got worse; no concrete symptom",
                             symptoms=[], specific=False,
                             missing=["What changed for you — is something failing, slower, or harder to use?",
                                      ASK_VERSION_EN],
                             user_terms=["app is bad"], dev_terms=[]),
            "reply": "Sorry to hear that — and thanks for telling us. Could you share what changed for you: is something failing, slower, or harder to use since an update? And which version are you on (Settings > About)? We read every one of these. — The Ritmo team",
        },
        # a support agent checking the change history before answering someone.
        # No reply is authored: a lookup has nobody waiting on one, and the
        # pipeline must not ask for a draft it would have no recipient for.
        "var-026": {
            "intake": intake(kind="question", summary="Asking whether the duplicate morning reminder was fixed",
                             symptoms=[], user_terms=["duplicate reminder"],
                             dev_terms=["notification", "reminder", "duplicate", "schedule"]),
            "lookup_triage": {"is_status_query": True,
                              "subject": "the same daily reminder notification arrives twice"},
            "adjudicate": A(NOTIF, 0.91, R_NOTIF, "duplicate reminder notifications"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="also save the authored outputs as replay fixtures")
    args = parser.parse_args()

    make_repo()
    index = RepoIndex(ROOT / "demo" / "app-repo", ROOT / "demo" / "releases.json")
    cases = [json.loads(l) for l in (ROOT / "evals" / "cases.jsonl").read_text().splitlines() if l.strip()]
    authored = build_cases(index)

    failures = []
    for case in cases:
        data = authored[case["id"]]
        item = FeedbackItem(id=case["id"], text=case["text"], channel=case.get("channel", "eval"))
        pipeline = Pipeline(index, AuthorLLM(data, write=args.write))
        result = pipeline.analyze(item)

        got, want = result.verdict.verdict.value, case["gold"]["verdict"]
        problems = []
        if got != want:
            problems.append(f"verdict {got} != gold {want}")
        if result.reply_is_fallback:
            problems.append("authored reply failed the version guardrail")
        gold_fixed_in = case["gold"].get("fixed_in")
        got_fixed_in = result.verdict.fix_commit.fixed_in if result.verdict.fix_commit else None
        if gold_fixed_in is not None and got_fixed_in != gold_fixed_in:
            problems.append(f"fixed_in {got_fixed_in} != gold {gold_fixed_in}")
        # retrieval sanity: the cited commit must have been in the candidate set
        if result.adjudication and result.adjudication.match_sha:
            in_cands = any(c.commit.sha == result.adjudication.match_sha for c in result.candidates)
            if not in_cands:
                problems.append("target commit was NOT retrieved into the candidate set")
        status = "ok " if not problems else "FAIL"
        print(f"  {case['id']:<10} {status} -> {got}" + ("  | " + "; ".join(problems) if problems else ""))
        if problems:
            failures.append((case["id"], problems))

    tail = ""
    if args.write:
        n_fixtures = len(list((ROOT / "fixtures" / "replay").glob("*.json")))
        tail = f"; {n_fixtures} fixtures written"
    print(f"\n{len(cases) - len(failures)}/{len(cases)} cases pass through the real pipeline{tail}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
