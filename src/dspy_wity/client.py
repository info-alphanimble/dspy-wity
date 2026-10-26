"""``Wity``: run DSPy decision outputs (``Noul``, ``Score``, ``Choice``) on Wity."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import dspy
import httpx
import wity
from dspy.adapters.decision import DecisionAdapter
from dspy.clients.base_lm import LM_CLASS_STATE_KEY, BaseLM, record_history
from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback, with_callbacks
from wity import Reasoning, SystemOneResponse

from ._update_notice import show_update_notice_once

MODEL = "wity-1"

# What DSPy reads from each answer. The rest (reasoning details, legend) stays in history.
_ANSWER_KEYS = ("noul", "score", "choice", "confidence", "probabilities")


class Wity(BaseLM):  # type: ignore[misc]  # dspy ships no type hints
    """Answer DSPy decision outputs with Wity, the typed-decision API.

    Attach it to a predictor whose outputs are all ``Noul``, ``Score`` or ``Choice``
    (or ``bool`` and ``Literal`` fields annotated with them)::

        triage = dspy.Predict(TriageTicket)
        triage.set_lm(Wity())

    Or set it for the whole program with ``dspy.configure(lm=Wity())``.
    Answers are cached in DSPy's cache, like any DSPy LM call.
    """

    supports_decision_requests = True

    def __init__(
        self,
        *,
        reasoning: Reasoning | None = None,
        max_latency_ms: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = wity.DEFAULT_TIMEOUT,
        max_retries: int = wity.DEFAULT_MAX_RETRIES,
        cache: bool = True,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
        callbacks: list[BaseCallback] | None = None,
    ) -> None:
        """
        Args:
            reasoning: ``"off"``, ``"auto"`` or ``"always"``. Leave out for the API default, ``"auto"``.
            max_latency_ms: Time limit per request, a whole number from 200 to 120000.
            api_key: Defaults to the WITY_API_KEY environment variable.
            base_url: Defaults to WITY_BASE_URL, then the SDK's default address.
            timeout: Seconds per attempt.
            max_retries: Retries on 408, 429, 5xx and network failures.
            cache: Whether to use DSPy's cache.
            http_client: Your own ``httpx.Client``, for example for a proxy.
            http_async_client: Your own ``httpx.AsyncClient`` for async calls.
            callbacks: DSPy callbacks for this client.
        """
        super().__init__(MODEL, cache=cache, callbacks=callbacks)
        self.reasoning: Reasoning | None = reasoning
        self.max_latency_ms = max_latency_ms
        # Kept for copy() and the async clients. Not in repr() or saved state.
        self._api_key = api_key
        self._http_client = http_client
        self._http_async_client = http_async_client
        self._options: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        hide_sdk_notice()
        # Made here, so a missing key or a bad address fails now, not at the first call.
        self._client = wity.WityClient(**self._options, http_client=http_client)
        show_update_notice_once("dspy-wity", "pip install -U dspy-wity")
        self.base_url = self._client.base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def __repr__(self) -> str:
        return f"Wity(reasoning={self.reasoning!r}, base_url={self.base_url!r})"

    @with_callbacks  # type: ignore[untyped-decorator]
    def __call__(self, state: Any, questions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Ask Wity the questions DSPy built for a predictor. DSPy calls this; you don't need to."""
        request = self._request(state, questions)
        response: dict[str, Any] | None = dspy.cache.get(request) if self.cache else None
        cache_hit = response is not None
        if response is None:
            answer = self._client.system_one(
                state=request["state"],
                questions=request["questions"],
                reasoning=self.reasoning,
                max_latency_ms=self.max_latency_ms,
            )
            response = _to_dict(answer)
        return self._finish(request, response, cache_hit)

    @with_callbacks  # type: ignore[untyped-decorator]
    async def acall(self, state: Any, questions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Async ``__call__``."""
        request = self._request(state, questions)
        response: dict[str, Any] | None = dspy.cache.get(request) if self.cache else None
        cache_hit = response is not None
        if response is None:
            # A new client per call: a connection can't be shared across event loops, and this way none is left open.
            async with wity.AsyncWityClient(**self._options, http_client=self._http_async_client) as client:
                answer = await client.system_one(
                    state=request["state"],
                    questions=request["questions"],
                    reasoning=self.reasoning,
                    max_latency_ms=self.max_latency_ms,
                )
            response = _to_dict(answer)
        return self._finish(request, response, cache_hit)

    def _request(self, state: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
        # Also the cache key, so everything that changes the answer is in it. Never the key.
        return {
            "provider": "wity",
            "base_url": self.base_url,
            "reasoning": self.reasoning,
            "max_latency_ms": self.max_latency_ms,
            "state": state,
            "questions": {name: _to_wity_question(name, q) for name, q in questions.items()},
        }

    def _finish(self, request: dict[str, Any], response: dict[str, Any], cache_hit: bool) -> dict[str, dict[str, Any]]:
        if self.cache and not cache_hit:
            dspy.cache.put(request, response)
        usage = {} if cache_hit else _usage(response)
        if settings.usage_tracker and usage:
            settings.usage_tracker.add_usage(response["model"], usage)
        answers = {
            name: {key: answer[key] for key in _ANSWER_KEYS if key in answer}
            for name, answer in response["answers"].items()
        }
        if not settings.disable_history:
            record_history(
                self,
                {
                    "request": copy.deepcopy(request),
                    "response": copy.deepcopy(response),
                    "usage": usage,
                    "cache_hit": cache_hit,
                    "model": response["model"],
                    "messages": None,
                    "prompt": json.dumps({"state": request["state"], "questions": request["questions"]}),
                    "outputs": [json.dumps(answers)],
                },
            )
        return answers

    def close(self) -> None:
        """Close the connections sync calls use. A client you passed in stays open."""
        self._client.close()

    def copy(self, **kwargs: Any) -> Wity:
        """A new ``Wity`` with the same settings and callbacks, and empty history."""
        options: dict[str, Any] = {
            "reasoning": self.reasoning,
            "max_latency_ms": self.max_latency_ms,
            "api_key": self._api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "cache": self.cache,
            "http_client": self._http_client,
            "http_async_client": self._http_async_client,
            "callbacks": self.callbacks,
        }
        unknown = sorted(kwargs.keys() - options.keys())
        if unknown:
            # dspy.BestOfN and dspy.Refine ask for rollout_id and temperature here, to sample new answers.
            raise TypeError(
                f"Wity.copy() got {unknown}. Wity gives one answer per question and doesn't sample, "
                "so modules that pick from several samples, like dspy.BestOfN and dspy.Refine, don't fit it."
            )
        return type(self)(**{**options, **kwargs})

    def __getstate__(self) -> dict[str, Any]:
        # For pickling, as `program.save(path, save_program=True)` does. Settings and callbacks only:
        # connections can't be pickled, and the key never goes into a file. The new process reads WITY_API_KEY.
        state = self.dump_state()
        state.pop(LM_CLASS_STATE_KEY)
        state["callbacks"] = self.callbacks
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state)  # type: ignore[misc]

    def __deepcopy__(self, memo: dict[int, Any]) -> Wity:
        # Open connections can't be copied. Optimizers deep-copy programs, and without this DSPy
        # warns, then shares the client anyway. Use copy() for a separate client.
        return self

    def dump_state(self) -> dict[str, Any]:
        """Settings to rebuild this client when a program is saved. Never the key."""
        return {
            LM_CLASS_STATE_KEY: f"{type(self).__module__}.{type(self).__qualname__}",
            "reasoning": self.reasoning,
            "max_latency_ms": self.max_latency_ms,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "cache": self.cache,
        }

    @classmethod
    def load_state(cls, state: dict[str, Any], **_: Any) -> Wity:
        """Rebuild a client from ``dump_state``. The key comes from WITY_API_KEY."""
        state = dict(state)
        state.pop(LM_CLASS_STATE_KEY, None)
        return cls(**state)


def _to_wity_question(name: str, question: Mapping[str, Any]) -> dict[str, Any]:
    """DSPy's question, changed only where Wity needs plain text."""
    kind = question.get("type")
    out: dict[str, Any] = {"type": kind, "instructions": _text(question.get("instructions"))}
    if "criteria" not in question:
        return out
    criteria = question["criteria"]
    if kind == "choice":
        # An option declared without a description is described by its label.
        out["criteria"] = {label: label if desc is None else _text(desc) for label, desc in criteria.items()}
    elif kind == "score":
        out["criteria"] = [_text(level) for level in criteria]
    elif kind == "noul":
        sides = {side: desc for side, desc in (criteria or {}).items() if desc is not None}
        if sides and sides.keys() != {"true", "false"}:
            raise ValueError(
                f"Wity needs criteria for both sides of {name!r}, 'true' and 'false', or neither. Got {sorted(sides)}."
            )
        if sides:
            out["criteria"] = {"true": _text(sides["true"]), "false": _text(sides["false"])}
    else:
        out["criteria"] = criteria
    return out


def _text(value: Any) -> Any:
    """Wity takes text. DSPy also allows objects and lists, so those are sent as JSON text."""
    return value if isinstance(value, str) or value is None else json.dumps(value, ensure_ascii=False)


def _to_dict(response: SystemOneResponse) -> dict[str, Any]:
    dumped: dict[str, Any] = response.model_dump(mode="json")
    return dumped


def _usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage") or {}
    return {"prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": usage.get("output_tokens", 0)}


# DSPy checks generation settings before it calls the client, and its message names another backend.
# Say "Wity" to Wity users instead. Drop this once DSPy's message is fixed upstream.
_dspy_call = DecisionAdapter.__call__
_dspy_acall = DecisionAdapter.acall


@contextmanager
def _settings_error_names_wity(lm: Any) -> Iterator[None]:
    try:
        yield
    except ValueError as err:
        message = str(err)
        if isinstance(lm, Wity) and message.startswith("Unsupported TypeSafe generation settings"):
            raise ValueError(
                message.replace("TypeSafe", "Wity", 1) + " Wity takes no generation settings like temperature."
            ) from None
        raise


def _call(self: Any, lm: Any, *args: Any, **kwargs: Any) -> Any:
    with _settings_error_names_wity(lm):
        return _dspy_call(self, lm, *args, **kwargs)


async def _acall(self: Any, lm: Any, *args: Any, **kwargs: Any) -> Any:
    with _settings_error_names_wity(lm):
        return await _dspy_acall(self, lm, *args, **kwargs)


DecisionAdapter.__call__ = _call
DecisionAdapter.acall = _acall


def hide_sdk_notice() -> None:
    """Keep wity-sdk's own notice quiet. It says to upgrade wity-sdk alone, which could pass the version we pin."""
    try:
        from wity._update_notice import _shown  # type: ignore[import-not-found,unused-ignore]  # wity-sdk 0.2.1+
    except ImportError:
        return
    _shown.add("wity-sdk")
