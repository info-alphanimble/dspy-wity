"""Route a support ticket with a DSPy signature answered by Wity.

Run: WITY_API_KEY=wity_... uv run python examples/quick_start.py
"""

import dspy
from dspy.experimental import Choice, Noul, Score

from dspy_wity import Wity

Team = Choice[
    ("billing", "Payments, charges and refunds"),
    ("technical", "Bugs, errors and outages"),
    ("other", "Anything else"),
]
Urgent = Noul[(True, "They ask for fast action"), (False, "No time pressure")]
Severity = Score["Not at all", "A little", "A lot"]


class Triage(dspy.Signature):
    """Route a customer support ticket."""

    ticket: str = dspy.InputField(desc="The customer's message.")
    team: Team = dspy.OutputField(desc="Which team should handle this ticket?")
    urgent: Urgent = dspy.OutputField(desc="Does the customer ask for something to happen soon?")
    severity: Severity = dspy.OutputField(desc="How much is this hurting the customer?")


dspy.configure(lm=Wity())
triage = dspy.Predict(Triage)

res = triage(ticket="I was charged twice for my subscription this month. Please refund one of them today.")

print(res.team.value)  # "billing"
print(res.team.probabilities)  # {"billing": 0.99..., "technical": ..., "other": ...}
print(res.urgent.value)  # True
print(res.urgent.probability)  # probability of yes
print(res.severity.value)  # expected level, from 0 (Not at all) to 2 (A lot)
print(res.severity.level)  # 2
