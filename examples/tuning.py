"""Tune when Wity's answer counts as "yes": by hand, then with ReAnchor.

Run: WITY_API_KEY=wity_... uv run python examples/tuning.py
"""

from typing import Any

import dspy
from dspy.experimental import Noul, ReAnchor

from dspy_wity import Wity

Refund = Noul[(True, "They ask for money back"), (False, "They don't ask for money back")]


class RefundCheck(dspy.Signature):
    """Read a customer support ticket."""

    ticket: str = dspy.InputField(desc="The customer's message.")
    refund: Refund = dspy.OutputField(desc="Does the customer ask for a refund?")


check = dspy.Predict(RefundCheck)
check.set_lm(Wity(reasoning="off"))

# By hand: only say yes when Wity is very sure.
check.fields["refund"] = {"threshold": 0.9}
res = check(ticket="The app charged me twice. Can you sort it out?")
print(res.refund.value, res.refund.probability)

# With ReAnchor: fit the threshold to labeled tickets.
tickets = {
    "Please refund my last payment, I cancelled before renewal.": True,
    "I want my money back for this month.": True,
    "You charged me twice, give one back.": True,
    "Can I get a refund for the unused days?": True,
    "Cancel my plan and return what I paid for October.": True,
    "Charge me for the annual plan instead of monthly.": False,
    "How do I download my invoice?": False,
    "The export button does nothing.": False,
    "Change my card on file please.": False,
    "Is there a student discount?": False,
}
trainset = [dspy.Example(ticket=t, refund=label).with_inputs("ticket") for t, label in tickets.items()]


def metric(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
    return float(pred.refund.value == example.refund)


check.fields = {}
optimizer = ReAnchor(metric)
tuned = optimizer.compile(check, trainset=trainset)
print(tuned.fields)  # the fitted threshold, or {} when the default already scores best
print(optimizer.report)  # the metric before and after
