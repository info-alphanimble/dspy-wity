"""``wity_tools``: Wity's split and extract as ``dspy.Tool`` objects, for agents like ``dspy.ReAct``."""

from __future__ import annotations

import dspy
import httpx
import wity

from ._update_notice import show_update_notice_once
from .client import hide_sdk_notice

_SPLIT_DESC = (
    "Cut a document into whole items, like emails in a thread, clauses in a contract or entries in a log. "
    "Returns the items in order, each exactly as written in the document."
)
_EXTRACT_DESC = (
    "Pull named fields out of a document, like names, dates, amounts or parties. "
    "Returns each field's values exactly as written, or an empty list when the document has none."
)


def wity_tools(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = wity.DEFAULT_TIMEOUT,
    max_retries: int = wity.DEFAULT_MAX_RETRIES,
    http_client: httpx.Client | None = None,
) -> list[dspy.Tool]:
    """Two tools an agent can call: ``wity_split`` and ``wity_extract``.

    ::

        agent = dspy.ReAct(ReadThread, tools=[*wity_tools(), other_tool])

    Both share one connection to Wity. A failed call raises, and DSPy's agents show the error to the model.

    Args:
        api_key: Defaults to the WITY_API_KEY environment variable.
        base_url: Defaults to WITY_BASE_URL, then the SDK's default address.
        timeout: Seconds per attempt. Split and extract wait at least two minutes, even with a shorter timeout.
        max_retries: Retries on 408, 429, 5xx and network failures.
        http_client: Your own ``httpx.Client``, for example for a proxy.
    """
    hide_sdk_notice()
    # Made here, so a missing key or a bad address fails now, not at the first call.
    client = wity.WityClient(
        api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries, http_client=http_client
    )
    show_update_notice_once("dspy-wity", "pip install -U dspy-wity")

    def wity_split(text: str, split: str | None = None) -> list[str]:
        return [item.text for item in client.split(text=text, split=split).items]

    def wity_extract(text: str, fields: dict[str, str]) -> dict[str, list[str]]:
        return client.extract(text=text, fields=fields).fields

    return [
        dspy.Tool(
            wity_split,
            name="wity_split",
            desc=_SPLIT_DESC,
            arg_desc={
                "text": "The whole document, up to 60,000 characters.",
                "split": 'What one item is, in plain words, like "one item per email". Leave out for paragraphs.',
            },
        ),
        dspy.Tool(
            wity_extract,
            name="wity_extract",
            desc=_EXTRACT_DESC,
            arg_desc={
                "text": "The whole document, up to 60,000 characters.",
                "fields": 'Field name to what it asks for, like {"party": "Who signs"}. 1 to 16 fields.',
            },
        ),
    ]
