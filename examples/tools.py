"""Call Wity's split and extract tools directly, as an agent would. Run: uv run python examples/tools.py"""

from dspy_wity import wity_tools

THREAD = """From: dana@example.com
The seal test on batch 2231 failed at 40 psi. Can we rerun it before Friday?

From: marcus@example.com
Yes. I booked the rig for Thursday 9 October at 10:00. It takes about 3 hours.
"""

split, extract = wity_tools()

for email in split(text=THREAD, split="one item per email"):
    print(extract(text=email, fields={"sender": "The email address the email is from", "asks": "What it asks for"}))
