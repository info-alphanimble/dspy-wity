"""wity_tools: the split and extract tools, on their own and inside dspy.ReAct."""

from __future__ import annotations

from typing import Any

import dspy
import httpx
import pytest
import wity
from dspy.utils.dummies import DummyLM

from dspy_wity import wity_tools

from .conftest import TEST_KEY, FakeWity

EMAIL = "From: dana@example.com\nSeal test failed.\n\nFrom: marcus@example.com\nRerun tomorrow."
SPLIT_REPLY: dict[str, Any] = {
    "items": [
        {"start": 0, "end": 40, "text": EMAIL[0:40]},
        {"start": 42, "end": len(EMAIL), "text": EMAIL[42:]},
    ],
    "how": "model",
    "timing_ms": {"total": 84},
    "usage": {"input_tokens": 412, "output_tokens": 0},
}
EXTRACT_REPLY: dict[str, Any] = {
    "fields": {"sender": ["dana@example.com", "marcus@example.com"]},
    "entities": [
        {"field": "sender", "text": "dana@example.com", "start": 6, "end": 22, "confidence": 0.98},
        {"field": "sender", "text": "marcus@example.com", "start": 48, "end": 66, "confidence": 0.97},
    ],
    "timing_ms": 503,
    "usage": {"input_tokens": 301, "output_tokens": 0},
}


@pytest.fixture
def fake() -> FakeWity:
    return FakeWity(default=httpx.Response(200, json=SPLIT_REPLY))


def tools_by_name(fake: FakeWity) -> dict[str, dspy.Tool]:
    return {tool.name: tool for tool in fake.tools()}


def test_two_tools_with_names_and_argument_schemas(fake: FakeWity) -> None:
    tools = tools_by_name(fake)
    assert set(tools) == {"wity_split", "wity_extract"}
    assert set(tools["wity_split"].args) == {"text", "split"}
    assert set(tools["wity_extract"].args) == {"text", "fields"}
    assert "whole items" in (tools["wity_split"].desc or "")


def test_split_returns_the_items_as_text(fake: FakeWity) -> None:
    items = tools_by_name(fake)["wity_split"](text=EMAIL, split="one item per email")
    assert items == [EMAIL[0:40], EMAIL[42:]]
    assert fake.requests[0].url.path == "/v1/split"
    assert fake.bodies[0] == {"text": EMAIL, "split": "one item per email"}


def test_split_without_a_split_leaves_it_to_the_api(fake: FakeWity) -> None:
    tools_by_name(fake)["wity_split"](text=EMAIL)
    assert fake.bodies[0] == {"text": EMAIL}


def test_extract_returns_each_fields_values(fake: FakeWity) -> None:
    fake.json(EXTRACT_REPLY)
    fields = {"sender": "Who each email is from"}
    values = tools_by_name(fake)["wity_extract"](text=EMAIL, fields=fields)
    assert values == {"sender": ["dana@example.com", "marcus@example.com"]}
    assert fake.requests[0].url.path == "/v1/extract"
    assert fake.bodies[0] == {"text": EMAIL, "fields": fields}


async def test_tools_work_in_the_async_path(fake: FakeWity) -> None:
    items = await tools_by_name(fake)["wity_split"].acall(text=EMAIL)
    assert len(items) == 2


def test_a_failed_call_raises_the_sdk_error(fake: FakeWity) -> None:
    fake.json({"error": "bad key"}, status=401)
    with pytest.raises(wity.AuthenticationError) as caught:
        tools_by_name(fake)["wity_split"](text=EMAIL)
    assert TEST_KEY not in str(caught.value)


def test_no_key_fails_when_the_tools_are_made() -> None:
    with pytest.raises(wity.WityError):
        wity_tools()


class ReadThread(dspy.Signature):
    """Say who wrote the emails in a thread."""

    thread: str = dspy.InputField()
    senders: list[str] = dspy.OutputField()


def test_react_agent_calls_the_tools(fake: FakeWity) -> None:
    fake.json(EXTRACT_REPLY)
    lm = DummyLM(
        [
            {
                "next_thought": "Pull out the senders.",
                "next_tool_name": "wity_extract",
                "next_tool_args": {"text": EMAIL, "fields": {"sender": "Who each email is from"}},
            },
            {"next_thought": "Done.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "Wity found two senders.", "senders": ["dana@example.com", "marcus@example.com"]},
        ]
    )
    agent = dspy.ReAct(ReadThread, tools=fake.tools())
    with dspy.context(lm=lm):
        result = agent(thread=EMAIL)
    assert result.senders == ["dana@example.com", "marcus@example.com"]
    assert fake.requests[0].url.path == "/v1/extract"
    assert "marcus@example.com" in str(result.trajectory["observation_0"])
