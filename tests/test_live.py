"""Real calls to Wity. Skipped unless WITY_LIVE=1. 8 calls, costing a fraction of a cent.

Needs WITY_API_KEY. See CONTRIBUTING.md for how to run them.
"""

import os
from typing import Any

import dspy
import pytest
import wity
from dspy.experimental import Noul, ReAnchor

from dspy_wity import Wity

from .test_client import TICKET, Triage

# Read before the clean_env fixture clears the environment for each test.
KEY = os.environ.get("WITY_API_KEY")
BASE_URL = os.environ.get("WITY_BASE_URL")

pytestmark = [
    pytest.mark.skipif(os.environ.get("WITY_LIVE") != "1", reason="set WITY_LIVE=1 to call the real API"),
    pytest.mark.enable_socket,
]


def live(**options: Any) -> Wity:
    return Wity(api_key=KEY, base_url=BASE_URL, cache=False, **options)


def triage(lm: Wity) -> dspy.Predict:
    predict = dspy.Predict(Triage)
    predict.set_lm(lm)
    return predict


def test_each_type() -> None:
    lm = live(reasoning="off")
    with dspy.context(track_usage=True):
        res = triage(lm)(ticket=TICKET)
    assert res.team.value == "billing"
    assert sum(res.team.probabilities.values()) == pytest.approx(1)
    assert res.urgent.value is True
    assert res.severity.level == 2
    assert set(res.severity.probabilities) == {0, 1, 2}
    assert res.get_lm_usage()["wity-1"]["prompt_tokens"] > 0
    assert lm.history[0]["response"]["model"] == "wity-1"


async def test_async_with_reasoning() -> None:
    lm = live(reasoning="always", max_latency_ms=5000)
    res = await triage(lm).acall(ticket=TICKET)
    assert res.team.value == "billing"
    assert lm.history[0]["response"]["answers"]["team"]["reasoning"]["mode"] == "always"


def test_object_instructions_are_accepted() -> None:
    # Wity rejects object instructions. The client sends them as JSON text.
    predict = triage(live(reasoning="off"))
    predict.fields["urgent"] = {"instructions": {"ask": "Does the customer ask for fast action?"}}
    assert predict(ticket=TICKET).urgent.value is True


def test_bad_key() -> None:
    with pytest.raises(wity.AuthenticationError):
        triage(Wity(api_key="wity_not_a_real_key", base_url=BASE_URL, cache=False, max_retries=0))(ticket=TICKET)


class Refund(dspy.Signature):
    """Read a customer support ticket."""

    ticket: str = dspy.InputField(desc="The customer's message.")
    refund: Noul = dspy.OutputField(desc="Does the customer ask for a refund?")


def test_reanchor() -> None:
    tickets = {
        "I want my money back for this month.": True,
        "Please refund the double charge.": True,
        "How do I download my invoice?": False,
        "The export button does nothing.": False,
    }
    predict = dspy.Predict(Refund)
    # ReAnchor needs the cache, so it can try many thresholds on the same answers.
    predict.set_lm(Wity(api_key=KEY, base_url=BASE_URL, reasoning="off"))
    trainset = [dspy.Example(ticket=t, refund=label).with_inputs("ticket") for t, label in tickets.items()]

    def metric(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
        return float(pred.refund.value == example.refund)

    optimizer = ReAnchor(metric, num_threads=1)
    optimizer.compile(predict, trainset=trainset)
    assert optimizer.report["train_score"] == 1.0
