"""A DSPy metric that asks Wity a yes/no question about each answer.

Pass it to dspy.Evaluate or to any optimizer that takes a metric.

Run: WITY_API_KEY=wity_... uv run python examples/metric.py
"""

from typing import Any

import dspy
from dspy.experimental import Noul

from dspy_wity import Wity

Helpful = Noul[(True, "The reply solves the customer's problem"), (False, "It doesn't, or it dodges")]


class JudgeReply(dspy.Signature):
    """Judge a support agent's reply to a customer."""

    question: str = dspy.InputField(desc="What the customer asked.")
    reply: str = dspy.InputField(desc="The agent's reply.")
    helpful: Helpful = dspy.OutputField(desc="Does the reply solve the customer's problem?")


judge = dspy.Predict(JudgeReply)
judge.set_lm(Wity(reasoning="off"))


def reply_is_helpful(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
    """Wity's probability that the reply helps. While an optimizer bootstraps (trace is set), a strict pass or fail."""
    verdict = judge(question=example.question, reply=pred.reply).helpful
    probability: float = verdict.probability
    return probability if trace is None else float(verdict.value)


question = dspy.Example(question="How do I reset my password?").with_inputs("question")
good = dspy.Prediction(reply="Click 'Forgot password' on the sign-in page and follow the email we send you.")
bad = dspy.Prediction(reply="Passwords are important. Thanks for reaching out!")

print(reply_is_helpful(question, good))  # close to 1
print(reply_is_helpful(question, bad))  # close to 0
