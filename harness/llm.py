"""One thin seam between the pipeline and the model (OpenAI GPT-5 family).

Three modes:
  live   — calls the OpenAI API (default when OPENAI_API_KEY resolves)
  replay — serves recorded/authored fixtures; lets the whole demo run keyless
  record — live + writes every response to fixtures/ (LANDED_RECORD=1)

Structured calls go through chat.completions.parse() with a Pydantic model, so
schema enforcement happens at the API layer, not in regex-land. GPT-5 models
are reasoning models: no temperature/top_p — behavior is steered by prompts
plus a per-call reasoning effort. Fixture keys hash the prompt content (not
the model), so switching models re-records cleanly.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "replay"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(Exception):
    """Transport/service failure after SDK retries. Pipeline degrades, never guesses."""


class LLMBadOutput(Exception):
    """Model refused or produced unusable output for a structured call."""


class ReplayMiss(Exception):
    """Replay mode has no fixture for this exact prompt."""


def _key(name: str, system: str, user: str) -> str:
    return hashlib.sha256(f"{name}\n{system}\n{user}".encode()).hexdigest()[:16]


class LLM:
    def __init__(self, mode: str | None = None):
        self.record = os.environ.get("LANDED_RECORD") == "1"
        self.client = None
        if mode:
            self.mode = mode
        elif os.environ.get("LANDED_REPLAY") == "1" and not self.record:
            self.mode = "replay"
        else:
            self.mode = "live" if self._try_client() else "replay"
        if self.mode == "live" and self.client is None and not self._try_client():
            raise LLMUnavailable("live mode requested but no OpenAI credentials resolve")

    def _try_client(self) -> bool:
        try:
            import openai
            client = openai.OpenAI()  # raises OpenAIError when no api_key resolves
        except Exception:
            self.client = None
            return False
        if getattr(client, "api_key", None):
            self.client = client
            return True
        self.client = None
        return False

    # ------------------------------------------------------------- calls

    def structured(self, name: str, system: str, user: str, model_cls: Type[T],
                   max_tokens: int = 4096, effort: str = "low") -> T:
        key = _key(name, system, user)
        if self.mode == "replay":
            return model_cls.model_validate(self._read_fixture(key, name))

        import openai
        try:
            completion = self.client.beta.chat.completions.parse(
                model=DEFAULT_MODEL,
                reasoning_effort=effort,
                max_completion_tokens=max_tokens,  # includes reasoning tokens
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format=model_cls,
            )
        except (openai.APIConnectionError, openai.APIStatusError) as e:
            raise LLMUnavailable(f"{name}: {type(e).__name__}") from e
        message = completion.choices[0].message
        if getattr(message, "refusal", None) or message.parsed is None:
            raise LLMBadOutput(f"{name}: refusal={getattr(message, 'refusal', None)!r}")
        result: T = message.parsed
        if self.record:
            self._write_fixture(key, name, "structured", result.model_dump())
        return result

    def text(self, name: str, system: str, user: str, max_tokens: int = 2048,
             effort: str = "low") -> str:
        key = _key(name, system, user)
        if self.mode == "replay":
            return str(self._read_fixture(key, name))

        import openai
        try:
            completion = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                reasoning_effort=effort,
                max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
        except (openai.APIConnectionError, openai.APIStatusError) as e:
            raise LLMUnavailable(f"{name}: {type(e).__name__}") from e
        message = completion.choices[0].message
        if getattr(message, "refusal", None) or not message.content:
            raise LLMBadOutput(f"{name}: refusal={getattr(message, 'refusal', None)!r}")
        out = message.content.strip()
        if self.record:
            self._write_fixture(key, name, "text", out)
        return out

    # ------------------------------------------------------------- fixtures

    def _read_fixture(self, key: str, name: str):
        path = FIXTURES_DIR / f"{key}.json"
        if not path.exists():
            raise ReplayMiss(
                f"no fixture for call '{name}' (key {key}). "
                "Run with OPENAI_API_KEY set, or LANDED_RECORD=1 to create it."
            )
        return json.loads(path.read_text())["output"]

    def _write_fixture(self, key: str, name: str, kind: str, output) -> None:
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        (FIXTURES_DIR / f"{key}.json").write_text(json.dumps(
            {"name": name, "kind": kind, "model": DEFAULT_MODEL,
             "authored": False, "output": output},
            ensure_ascii=False, indent=1,
        ))


def write_authored_fixture(name: str, system: str, user: str, output) -> str:
    """Used by scripts/author_fixtures.py to pre-seed replay outputs by hand."""
    key = _key(name, system, user)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / f"{key}.json").write_text(json.dumps(
        {"name": name, "kind": "authored", "model": "hand-authored",
         "authored": True, "output": output},
        ensure_ascii=False, indent=1,
    ))
    return key
