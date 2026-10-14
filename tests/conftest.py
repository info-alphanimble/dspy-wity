"""Shared test helpers: a fake Wity server on a fake network."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dspy
import httpx
import pytest

from dspy_wity import Wity

TEST_KEY = "wity_test_secret_1234567890"
"""A made-up key. Tests check that it never leaks."""

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_body(name: str) -> dict[str, Any]:
    """The body of a real API response, saved in tests/fixtures/."""
    saved: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text())
    body: dict[str, Any] = saved["body"]
    return body


def noul_answer(probability: float) -> dict[str, Any]:
    return {"type": "noul", "noul": probability}


def choice_answer(probabilities: dict[str, float]) -> dict[str, Any]:
    return {
        "type": "choice",
        "choice": max(probabilities, key=lambda k: probabilities[k]),
        "probabilities": probabilities,
        "confidence": 0.9,
    }


def score_answer(probabilities: list[float]) -> dict[str, Any]:
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probabilities)),
        "legend": {str(i): f"level {i}" for i in range(len(probabilities))},
        "probabilities": {str(i): p for i, p in enumerate(probabilities)},
        "confidence": 0.8,
    }


def body_with(**answers: dict[str, Any]) -> dict[str, Any]:
    """A real response body with these answers in it."""
    return {**fixture_body("multi-auto"), "answers": answers}


Reply = httpx.Response | Callable[[httpx.Request], httpx.Response] | Exception


@dataclass
class FakeWity:
    """Answers every request with the next queued reply, or ``default``. Records what was sent."""

    default: Reply = field(default_factory=lambda: httpx.Response(200, json=fixture_body("multi-auto")))
    queue: list[Reply] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def json(self, body: dict[str, Any], status: int = 200) -> None:
        self.queue.append(httpx.Response(status, json=body))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.queue.pop(0) if self.queue else self.default
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        return reply(request)

    def wity(self, **options: Any) -> Wity:
        """A ``Wity`` wired to this fake server. No retries and no cache unless asked for."""
        transport = httpx.MockTransport(self._handle)
        options.setdefault("api_key", TEST_KEY)
        options.setdefault("max_retries", 0)
        options.setdefault("cache", False)
        return Wity(
            http_client=httpx.Client(transport=transport),
            http_async_client=httpx.AsyncClient(transport=transport),
            **options,
        )


@pytest.fixture
def fake() -> FakeWity:
    return FakeWity()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests never see the developer's real key or address."""
    for name in ("WITY_API_KEY", "WITY_BASE_URL", "WITY_LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture(autouse=True)
def fresh_dspy() -> Iterator[None]:
    """Each test gets an empty in-memory DSPy cache and no global LM."""
    old_cache = dspy.cache
    dspy.cache = dspy.clients.Cache(enable_disk_cache=False, enable_memory_cache=True, disk_cache_dir=None)
    with dspy.context(lm=None):
        yield
    dspy.cache = old_cache
