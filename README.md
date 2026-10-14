# dspy-wity

Use [Wity](https://wity.alphanimble.com), the typed-decision API, in DSPy. Write a normal DSPy signature with yes/no, choice or score outputs, and Wity answers it with a probability for every option.

It's one class, `Wity`, that plugs in where DSPy expects an LM. It works with DSPy's own decision types (`Noul`, `Choice`, `Score`) and its `ReAnchor` optimizer.

## Install

```sh
pip install dspy-wity
```

Or with uv: `uv add dspy-wity`. Needs Python 3.10 or newer and DSPy 3.4 or newer, where the decision types live in `dspy.experimental`. It's built on the [`wity-sdk`](https://pypi.org/project/wity-sdk/) package.

## Quick start

Set your key in the environment:

```sh
export WITY_API_KEY="wity_..."
```

Then write a signature whose outputs are decisions:

```python
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

res.team.value             # "billing"
res.team.probabilities     # {"billing": 0.99954..., "technical": 0.00019..., "other": 0.00026...}
res.urgent.value           # True
res.urgent.probability     # 0.99995... = probability of yes
res.severity.value         # 1.97326... = expected level, from 0 (Not at all) to 2 (A lot)
res.severity.level         # 2
```

Each output's `desc` becomes Wity's question. The signature's docstring and inputs become what Wity reads. Probabilities sum to 1, but they aren't calibrated. Treat them as a ranking of how sure Wity is, not as exact odds.

To use Wity for one predictor only, attach it there: `triage.set_lm(Wity())`. Every output of that predictor must be a decision. Other predictors in the same program can keep using a chat model for text.

If you'd rather get plain Python values, annotate a `bool` or `Literal` with the decision type:

```python
from typing import Annotated, Literal

class PlainTriage(dspy.Signature):
    """Route a customer support ticket."""

    ticket: str = dspy.InputField(desc="The customer's message.")
    urgent: Annotated[bool, Urgent] = dspy.OutputField(desc="Does the customer ask for something to happen soon?")
    team: Annotated[Literal["billing", "technical", "other"], Team] = dspy.OutputField(desc="Which team?")


plain = dspy.Predict(PlainTriage)(ticket="I was charged twice. Please refund one today.")
plain.urgent   # True
plain.team     # "billing"
```

## Which DSPy modules fit

Wity answers questions with fixed options. It doesn't write text and gives one answer per question, so:

- **`dspy.Predict`** is the module to use. `dspy.Parallel`, `dspy.Evaluate` (with threads too), `dspy.asyncify` and your own `dspy.Module` programs work as usual.
- **`dspy.ChainOfThought`** adds a text `reasoning` output, which Wity can't write. Use `Predict` with `Wity(reasoning="always")` to have Wity think first.
- **`dspy.BestOfN`** and **`dspy.Refine`** pick from several sampled answers. Wity doesn't sample, so they raise an error. Read the probabilities instead.
- **`dspy.streamify`** isn't supported for decisions.
- **Optimizers:** `ReAnchor` keeps a predictor's `set_lm(Wity())`. `LabeledFewShot` and `BootstrapFewShot` start from a fresh copy of your program, which drops it, as with any LM. Set Wity for the whole run instead, with `dspy.configure(lm=Wity())` or `with dspy.context(lm=Wity()):`.

## Tuning when an answer counts

Wity gives the probabilities. DSPy turns them into a value with a setting per output: a threshold for `Noul` (default 0.5), cuts between levels for `Score`, and weights for `Choice`. Set them by hand:

```python
triage.fields["urgent"] = {"threshold": 0.9}                 # only say yes when Wity is very sure
triage.fields["severity"] = {"cuts": [0.8, 1.5]}             # where one level ends and the next begins
triage.fields["team"] = {"weights": {"billing": 2.0}}        # prefer billing in close calls
```

Or let `ReAnchor` fit them to labeled examples. It runs your examples once, then tries settings on the cached answers, so it makes one Wity call per example:

```python
from dspy.experimental import ReAnchor

trainset = [
    dspy.Example(ticket="I want my money back for this month.", urgent=True).with_inputs("ticket"),
    dspy.Example(ticket="How do I download my invoice?", urgent=False).with_inputs("ticket"),
    # ... more labeled tickets
]

def metric(example, pred, trace=None):
    return float(pred.urgent.value == example.urgent)

optimizer = ReAnchor(metric)
tuned = optimizer.compile(triage, trainset=trainset)
tuned.fields        # the fitted settings, or {} when the defaults already score best
optimizer.report    # the metric before and after
```

`ReAnchor` needs DSPy's cache, which `Wity` uses by default.

## Wity as a metric

DSPy's optimizers and `dspy.Evaluate` take a metric: a function that scores each answer. Wity can be that judge. Ask it a yes/no question about the answer, and use its probability as the score:

```python
Helpful = Noul[(True, "The reply solves the customer's problem"), (False, "It doesn't, or it dodges")]


class JudgeReply(dspy.Signature):
    """Judge a support agent's reply to a customer."""

    question: str = dspy.InputField(desc="What the customer asked.")
    reply: str = dspy.InputField(desc="The agent's reply.")
    helpful: Helpful = dspy.OutputField(desc="Does the reply solve the customer's problem?")


judge = dspy.Predict(JudgeReply)
judge.set_lm(Wity(reasoning="off"))


def reply_is_helpful(example, pred, trace=None):
    verdict = judge(question=example.question, reply=pred.reply).helpful
    return verdict.probability if trace is None else float(verdict.value)
```

Your program can run on any chat model while Wity judges it. Pass the metric on as usual, for example `dspy.Evaluate(devset=devset, metric=reply_is_helpful)` or `dspy.MIPROv2(metric=reply_is_helpful)`. While an optimizer collects examples (`trace` is set), the metric returns a strict pass or fail.

## Reasoning and speed

Wity answers most questions in one fast pass. With `reasoning="auto"` it thinks first when a question needs it, for example when the top options are close. That usually takes 1–4 seconds.

| Setting | What it does |
|---|---|
| `"off"` | Never thinks. Fastest. |
| `"auto"` | Thinks only when the question needs it. The API default. |
| `"always"` | Always thinks. |

```python
Wity(reasoning="off")
Wity(reasoning="always", max_latency_ms=5000)
```

`max_latency_ms=` (a whole number from 200 to 120000) caps the time per request. Thinking stops early to meet it.

## Options

| Option | Default | What it does |
|---|---|---|
| `reasoning` | API default (`"auto"`) | See above |
| `max_latency_ms` | none | Time limit per request |
| `api_key` | `WITY_API_KEY` | Your key |
| `base_url` | `WITY_BASE_URL`, then Wity's address | Where requests go |
| `timeout` | 60 | Seconds per attempt |
| `max_retries` | 2 | Retries on 408, 429, 5xx and network failures |
| `cache` | `True` | Use DSPy's cache |
| `http_client`, `http_async_client` | none | Your own `httpx` clients, for example for a proxy |
| `callbacks` | none | DSPy callbacks for this client |

`Wity` works in sync and async code (`await triage.acall(...)`), from many threads at once, and across event loops. Keep one around and reuse it, so sync calls reuse their connections. Call `close()` when you're done.

## Saving and loading

Save a tuned program as usual. The fitted settings are saved with it. Your key never is:

```python
tuned.save("triage.json")
```

To load it with `Wity` attached, pass DSPy's `allow_unsafe_lm_state=True`. DSPy asks for it before it loads any LM class from outside DSPy. Only pass it for files you trust:

```python
loaded = dspy.Predict(Triage)
loaded.load("triage.json", allow_unsafe_lm_state=True)   # the key comes from WITY_API_KEY
```

Or save the program without an LM, and attach `Wity()` after loading:

```python
tuned.set_lm(None)
tuned.save("triage.json")

loaded = dspy.Predict(Triage)
loaded.load("triage.json")
loaded.set_lm(Wity())
```

Saving the whole program (`save(path, save_program=True)`) and pickling work too. A pickled `Wity` keeps its settings, not its connections or your key. The process that loads it reads `WITY_API_KEY`.

## Errors, retries and the cache

**Errors** are the `wity-sdk` errors, unchanged:

| Error | When |
|---|---|
| `BadRequestError` | 400, for example a state that's too long |
| `AuthenticationError` | 401, a wrong key |
| `InsufficientBalanceError` | 402 |
| `RateLimitError` | 429 |
| `InternalServerError` | 5xx |
| `APIConnectionError`, `APITimeoutError` | Network problems |

```python
import wity

try:
    res = triage(ticket=ticket)
except wity.RateLimitError as err:
    print(err.status, err.headers.get("retry-after"))
```

Error messages and logs never include your state or your key.

**Retries.** `Wity` already retries failed calls (`max_retries=2`). DSPy doesn't retry them again.

**Cache.** Answers go in DSPy's cache, keyed by the question, the inputs and the options. The same call again comes from the cache, with no request and no usage. Turn it off with `Wity(cache=False)`.

**History and usage.** `lm.inspect_history()` shows the last calls, with Wity's full answers. With `dspy.configure(track_usage=True)`, `res.get_lm_usage()` counts Wity's input tokens under `wity-1`.

## Examples

The `examples/` folder in the repository has a script for the quick start, tuning and the metric. Each runs with only a Wity key:

```sh
export WITY_API_KEY="wity_..."
uv run python examples/quick_start.py
```

## Contributing

See `CONTRIBUTING.md` in the repository for how to run the checks and tests.

## License

Apache-2.0
