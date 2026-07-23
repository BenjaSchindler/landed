"""One thin seam between the pipeline and the model.

Three modes:
  live   — calls the Claude API (default when credentials resolve)
  replay — serves recorded/authored fixtures; lets the whole demo run keyless
  record — live + writes every response to fixtures/ (LANDED_RECORD=1)

Structured calls go through client.messages.parse() with a Pydantic model, so
schema enforcement happens at the API layer, not in regex-land. Fixture keys
hash the prompt content (not the model), so switching models re-records cleanly.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "replay"
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")

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
            raise LLMUnavailable("live mode requested but no Anthropic credentials resolve")

    def _try_client(self) -> bool:
        # Anthropic() constructs fine without credentials and only fails at
        # request time — so check that something actually resolved.
        try:
            import anthropic
            client = anthropic.Anthropic()
        except Exception:
            self.client = None
            return False
        if getattr(client, "api_key", None) or getattr(client, "auth_token", None):
            self.client = client
            return True
        self.client = None
        return False

    # ------------------------------------------------------------- calls

    def structured(self, name: str, system: str, user: str, model_cls: Type[T],
                   max_tokens: int = 4096, thinking: bool = False) -> T:
        key = _key(name, system, user)
        if self.mode == "replay":
            return model_cls.model_validate(self._read_fixture(key, name))

        import anthropic
        kwargs: dict = dict(
            model=DEFAULT_MODEL, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}], output_format=model_cls,
        )
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        try:
            response = self.client.messages.parse(**kwargs)
        except (anthropic.APIConnectionError, anthropic.APIStatusError, TypeError) as e:
            raise LLMUnavailable(f"{name}: {type(e).__name__}") from e
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise LLMBadOutput(f"{name}: stop_reason={response.stop_reason}")
        result: T = response.parsed_output
        if self.record:
            self._write_fixture(key, name, "structured", result.model_dump())
        return result

    def text(self, name: str, system: str, user: str, max_tokens: int = 1024) -> str:
        key = _key(name, system, user)
        if self.mode == "replay":
            return str(self._read_fixture(key, name))

        import anthropic
        try:
            response = self.client.messages.create(
                model=DEFAULT_MODEL, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
        except (anthropic.APIConnectionError, anthropic.APIStatusError, TypeError) as e:
            raise LLMUnavailable(f"{name}: {type(e).__name__}") from e
        if response.stop_reason == "refusal" or not response.content:
            raise LLMBadOutput(f"{name}: stop_reason={response.stop_reason}")
        out = "".join(b.text for b in response.content if b.type == "text").strip()
        if self.record:
            self._write_fixture(key, name, "text", out)
        return out

    # ------------------------------------------------------------- fixtures

    def _read_fixture(self, key: str, name: str):
        path = FIXTURES_DIR / f"{key}.json"
        if not path.exists():
            raise ReplayMiss(
                f"no fixture for call '{name}' (key {key}). "
                "Run with ANTHROPIC_API_KEY set, or LANDED_RECORD=1 to create it."
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
