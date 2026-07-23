"""Generate the demo repo: 'Ritmo', a habit & wellbeing app for a small
Chilean team that ships through the app stores.

The history is deterministic (fixed dates, author, content) so commit shas are
stable across machines — replay fixtures and eval gold labels depend on that.

The history is deliberately messy-realistic: good fix messages, terse ones,
noise commits, and near-miss baits (a streak *timezone* fix that must NOT match
a streak *layout* complaint). Landed itself works on any git repo; this one
exists so the demo and evals are self-contained.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent / "app-repo"
AUTHOR = "Ana Riquelme <ana@ritmo.app>"

# (date, subject, body, [files]) — tags interleaved as ("TAG", name)
HISTORY: list = [
    ("2026-02-10T09:12:00-03:00", "feat: project scaffold, navigation and theme",
     "", ["src/App.tsx", "src/theme/colors.ts", "src/navigation/root.tsx"]),
    ("2026-02-12T11:40:00-03:00", "feat(habits): habit list and daily check-off",
     "", ["src/screens/HabitsScreen.tsx", "src/habits/store.ts"]),
    ("2026-02-14T16:05:00-03:00", "feat(exercise): guided exercise runner with steps and answers",
     "Steps advance manually or on a timer; answers held in the runner state.",
     ["src/exercise/runner.ts", "src/screens/ExerciseScreen.tsx"]),
    ("2026-02-17T10:22:00-03:00", "feat(timer): breathing timer with inhale/exhale phases",
     "", ["src/timer/session.ts", "src/screens/BreathingScreen.tsx"]),
    ("2026-02-19T18:30:00-03:00", "feat(notifications): daily reminder scheduling",
     "", ["src/notifications/scheduler.ts"]),
    ("2026-02-20T12:00:00-03:00", "feat(profile): profile screen with avatar upload",
     "", ["src/screens/ProfileScreen.tsx", "src/profile/avatar.ts"]),
    ("2026-02-23T15:45:00-03:00", "feat(stats): stats page with weekly charts",
     "", ["src/screens/StatsScreen.tsx", "src/stats/charts.tsx"]),
    ("2026-02-25T09:10:00-03:00", "feat(streaks): daily streak counter",
     "", ["src/streaks/streaks.ts", "src/screens/HomeScreen.tsx"]),
    ("2026-02-26T14:20:00-03:00", "chore: app icons, splash, store metadata",
     "", ["assets/icon.png.txt", "store/listing-es.md"]),
    ("2026-02-27T17:00:00-03:00", "i18n: spanish and english strings",
     "", ["src/i18n/es.json", "src/i18n/en.json"]),
    ("TAG", "v1.0.0"),

    ("2026-03-05T10:30:00-03:00", "fix: onboarding typo and button copy",
     "", ["src/i18n/es.json"]),
    ("2026-03-09T16:15:00-03:00", "fix(profile): crash when uploading avatar photo on Android 13+",
     "READ_MEDIA_IMAGES permission was never requested on API 33+, so picking a\n"
     "profile photo from the gallery crashed the app. Request the granular\n"
     "media permission and fall back gracefully when denied.",
     ["src/profile/avatar.ts", "android/app/src/main/AndroidManifest.xml.txt"]),
    ("2026-03-12T11:00:00-03:00", "chore: bump react-native and gradle",
     "", ["package.json"]),
    ("2026-03-13T09:45:00-03:00", "fix stuff",
     "wrong padding on small screens in onboarding", ["src/screens/OnboardingScreen.tsx"]),
    ("TAG", "v1.0.1"),

    ("2026-03-24T10:00:00-03:00", "feat(exercise): reflection questions with free-text answers",
     "", ["src/exercise/questions.ts", "src/exercise/runner.ts"]),
    ("2026-04-02T15:30:00-03:00", "feat(timer): ambient sounds during breathing timer",
     "", ["src/timer/sounds.ts"]),
    ("2026-04-10T12:20:00-03:00", "refactor: extract exercise runner state machine",
     "", ["src/exercise/runner.ts", "src/exercise/state.ts"]),
    ("2026-04-16T17:40:00-03:00", "feat(stats): monthly view and mood overlay",
     "", ["src/screens/StatsScreen.tsx", "src/stats/mood.ts"]),
    ("2026-04-22T09:30:00-03:00", "fix(exercise): persist in-progress answers when the app is backgrounded",
     "Answers and step progress lived only in the runner's memory. Taking a\n"
     "phone call or switching to another app unmounted the runner and dropped\n"
     "everything the user had written. Flush runner state to storage on every\n"
     "AppState change to background and restore it on resume. Fixes #124.",
     ["src/exercise/runner.ts", "src/exercise/persist.ts"]),
    ("2026-04-28T14:10:00-03:00", "fix(notifications): cancel scheduled reminders before re-scheduling",
     "Scheduling on every app open queued a duplicate, so users received the\n"
     "same daily reminder two or more times. Cancel pending notifications for\n"
     "the habit before scheduling the next one. Fixes #131.",
     ["src/notifications/scheduler.ts"]),
    ("2026-05-02T11:25:00-03:00", "copy: reword exercise completion screen",
     "", ["src/i18n/es.json", "src/i18n/en.json"]),
    ("2026-05-05T16:50:00-03:00", "test: exercise runner persistence tests",
     "", ["src/exercise/__tests__/persist.test.ts"]),
    ("TAG", "v1.1.0"),

    ("2026-05-20T10:15:00-03:00", "feat(notifications): quiet hours",
     "", ["src/notifications/quiet.ts"]),
    ("2026-05-28T15:00:00-03:00", "style: improve dark mode contrast for secondary buttons",
     "", ["src/theme/colors.ts"]),
    ("2026-06-04T09:40:00-03:00", "fix(timer): breathing timer no longer restarts after switching apps",
     "The timer rebuilt its state from zero when the app regained focus,\n"
     "restarting the session mid-breath. Resume from a monotonic start\n"
     "timestamp instead of re-initializing on focus.",
     ["src/timer/session.ts"]),
    ("2026-06-12T13:30:00-03:00", "feat(habits): weekly habit goals",
     "", ["src/habits/goals.ts", "src/screens/HabitsScreen.tsx"]),
    ("2026-06-20T18:05:00-03:00", "fix(streaks): count days by local midnight, not UTC",
     "Users west of UTC lost their streak when completing a habit late at\n"
     "night: the day boundary was computed in UTC. Compute day keys in the\n"
     "device timezone.",
     ["src/streaks/streaks.ts"]),
    ("2026-06-27T10:50:00-03:00", "chore: analytics events for exercise completion",
     "", ["src/analytics/events.ts"]),
    ("2026-07-08T16:20:00-03:00", "feat(export): share a weekly summary image",
     "", ["src/export/summary.tsx"]),
    ("2026-07-15T11:10:00-03:00", "ci: build both store flavors on release branches",
     "", [".github/workflows/release.yml"]),
    ("TAG", "v1.2.0"),

    ("2026-07-18T09:35:00-03:00", "fix(stats): stats page slow to open with a year of history",
     "Charts recomputed every aggregate on each render; with 12+ months of\n"
     "entries opening the stats tab took seconds. Memoize the aggregation and\n"
     "compute it off the render path.",
     ["src/screens/StatsScreen.tsx", "src/stats/charts.tsx"]),
    ("2026-07-21T15:45:00-03:00", "wip: experiment with new home layout",
     "", ["src/screens/HomeScreen.tsx"]),
]


def run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def main() -> Path:
    if REPO.exists():
        shutil.rmtree(REPO)
    REPO.mkdir(parents=True)
    run("git", "init", "-q", "-b", "main", cwd=REPO)
    run("git", "config", "user.name", "Ana Riquelme", cwd=REPO)
    run("git", "config", "user.email", "ana@ritmo.app", cwd=REPO)

    import os
    for entry in HISTORY:
        if entry[0] == "TAG":
            run("git", "tag", entry[1], cwd=REPO)
            continue
        date, subject, body, files = entry
        for f in files:
            path = REPO / f
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fh:
                fh.write(f"// {date} {subject}\n")
        run("git", "add", "-A", cwd=REPO)
        env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
               "GIT_AUTHOR_NAME": "Ana Riquelme", "GIT_AUTHOR_EMAIL": "ana@ritmo.app",
               "GIT_COMMITTER_NAME": "Ana Riquelme", "GIT_COMMITTER_EMAIL": "ana@ritmo.app"}
        message = subject + (f"\n\n{body}" if body else "")
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=REPO, env=env,
                       check=True, capture_output=True)
    return REPO


if __name__ == "__main__":
    path = main()
    n = subprocess.run(["git", "-C", str(path), "rev-list", "--count", "HEAD"],
                       capture_output=True, text=True).stdout.strip()
    print(f"demo repo ready at {path} ({n} commits)")
