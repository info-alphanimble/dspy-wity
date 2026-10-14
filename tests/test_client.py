import json
import logging
import pickle
from pathlib import Path
from typing import Annotated, Any, Literal

import dspy
import httpx
import pytest
import wity
from dspy.experimental import Choice, Noul, Score
from dspy.utils.callback import BaseCallback

from dspy_wity import Wity

from .conftest import TEST_KEY, FakeWity, body_with, choice_answer, fixture_body, noul_answer, score_answer

Urgent = Noul[(True, "Needs action today"), (False, "Can wait")]
Severity = Score["Not at all", "A little", "A lot"]
Team = Choice[("billing", "Payments and refunds"), ("technical", "Bugs and outages"), ("other", "Anything else")]


class Triage(dspy.Signature):
    """Route a support ticket."""

    ticket: str = dspy.InputField(desc="The customer's message.")
    team: Team = dspy.OutputField(desc="Which team should handle this ticket?")
    urgent: Urgent = dspy.OutputField(desc="Does the customer ask for fast action?")
    severity: Severity = dspy.OutputField(desc="How much is this hurting the customer?")


TICKET = "I was charged twice for my subscription this month. Please refund one of them today."


def triage(lm: Wity) -> dspy.Predict:
    predict = dspy.Predict(Triage)
    predict.set_lm(lm)
    return predict


def only_urgent(probability: float) -> dict[str, Any]:
    return body_with(urgent=noul_answer(probability))


class UrgentOnly(dspy.Signature):
    """Check a support ticket."""

    ticket: str = dspy.InputField(desc="The customer's message.")
    urgent: Urgent = dspy.OutputField(desc="Does the customer ask for fast action?")


# --- answers ---------------------------------------------------------------


def test_each_type_comes_back_with_wity_numbers(fake: FakeWity) -> None:
    res = triage(fake.wity())(ticket=TICKET)
    answers = fixture_body("multi-auto")["answers"]

    assert res.team.value == "billing"
    assert res.team.probabilities == answers["team"]["probabilities"]
    assert res.team.confidence == answers["team"]["confidence"]

    assert res.urgent.value is True
    assert res.urgent.probability == answers["urgent"]["noul"]

    # DSPy turns the level keys into ints and works out the expected level itself.
    levels = {int(k): v for k, v in answers["severity"]["probabilities"].items()}
    assert res.severity.probabilities == levels
    assert res.severity.value == pytest.approx(answers["severity"]["score"])
    assert res.severity.level == 2


async def test_async(fake: FakeWity) -> None:
    res = await triage(fake.wity()).acall(ticket=TICKET)
    assert res.team.value == "billing"
    assert len(fake.requests) == 1


def test_native_annotations_give_plain_values(fake: FakeWity) -> None:
    class Plain(dspy.Signature):
        """Route a support ticket."""

        ticket: str = dspy.InputField(desc="The customer's message.")
        urgent: Annotated[bool, Urgent] = dspy.OutputField(desc="Does the customer ask for fast action?")
        team: Annotated[Literal["billing", "technical", "other"], Team] = dspy.OutputField(desc="Which team?")

    predict = dspy.Predict(Plain)
    predict.set_lm(fake.wity())
    res = predict(ticket=TICKET)
    assert res.urgent is True
    assert res.team == "billing"


def test_field_settings_change_the_pick(fake: FakeWity) -> None:
    predict = dspy.Predict(UrgentOnly)
    predict.set_lm(fake.wity())
    fake.default = httpx.Response(200, json=only_urgent(0.6))

    assert predict(ticket=TICKET).urgent.value is True
    predict.fields["urgent"] = {"threshold": 0.7}
    res = predict(ticket=TICKET)
    assert res.urgent.value is False
    assert res.urgent.probability == 0.6


def test_choice_weights_change_the_pick(fake: FakeWity) -> None:
    predict = triage(fake.wity())
    predict.fields["team"] = {"weights": {"billing": 0.00001}}
    assert predict(ticket=TICKET).team.value == "other"


def test_unknown_answer_type_is_dspys_error(fake: FakeWity) -> None:
    fake.json(body_with(urgent={"type": "something_new", "value": 1}))
    predict = dspy.Predict(UrgentOnly)
    predict.set_lm(fake.wity())
    with pytest.raises(KeyError):
        predict(ticket=TICKET)


# --- what is sent ----------------------------------------------------------


def test_request_body(fake: FakeWity) -> None:
    triage(fake.wity(reasoning="off", max_latency_ms=800))(ticket=TICKET)
    body = fake.bodies[0]
    assert body["reasoning"] == "off"
    assert body["max_latency_ms"] == 800
    assert body["state"] == {
        "instructions": "Route a support ticket.",
        "input_fields": "1. `ticket` (str): The customer's message.",
        "inputs": {"ticket": TICKET},
    }
    assert body["questions"] == {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments and refunds",
                "technical": "Bugs and outages",
                "other": "Anything else",
            },
        },
        "urgent": {
            "type": "noul",
            "instructions": "Does the customer ask for fast action?",
            "criteria": {"true": "Needs action today", "false": "Can wait"},
        },
        "severity": {
            "type": "score",
            "instructions": "How much is this hurting the customer?",
            "criteria": ["Not at all", "A little", "A lot"],
        },
    }


def test_defaults_leave_reasoning_to_the_api(fake: FakeWity) -> None:
    triage(fake.wity())(ticket=TICKET)
    assert "reasoning" not in fake.bodies[0]
    assert "max_latency_ms" not in fake.bodies[0]


def test_key_goes_in_the_header(fake: FakeWity) -> None:
    triage(fake.wity())(ticket=TICKET)
    assert fake.requests[0].headers["authorization"] == f"Bearer {TEST_KEY}"
    assert str(fake.requests[0].url) == "https://api.wity.alphanimble.com/v1/systemone"


def test_demos_go_in_the_state(fake: FakeWity) -> None:
    predict = triage(fake.wity())
    predict.demos = [dspy.Example(ticket="App crashes", team="technical", urgent=True, severity=2)]
    predict(ticket=TICKET)
    assert fake.bodies[0]["state"]["demos"] == [
        {"ticket": "App crashes", "team": "technical", "urgent": True, "severity": 2}
    ]


def test_bare_noul_sends_no_criteria(fake: FakeWity) -> None:
    class Bare(dspy.Signature):
        """Check a support ticket."""

        ticket: str = dspy.InputField()
        urgent: Noul = dspy.OutputField(desc="Is it urgent?")

    fake.default = httpx.Response(200, json=only_urgent(0.9))
    predict = dspy.Predict(Bare)
    predict.set_lm(fake.wity())
    assert predict(ticket=TICKET).urgent.value is True
    assert fake.bodies[0]["questions"]["urgent"] == {"type": "noul", "instructions": "Is it urgent?"}


def test_object_instructions_and_criteria_become_json_text(fake: FakeWity) -> None:
    predict = triage(fake.wity())
    predict.fields["urgent"] = {"instructions": {"ask": "Is it urgent?", "for": "support"}}
    predict.set_criteria("severity", ["None", {"label": "Some"}, "Lots"])
    predict(ticket=TICKET)
    questions = fake.bodies[0]["questions"]
    assert questions["urgent"]["instructions"] == '{"ask": "Is it urgent?", "for": "support"}'
    assert questions["severity"]["criteria"] == ["None", '{"label": "Some"}', "Lots"]


def test_choice_option_without_description_uses_its_label(fake: FakeWity) -> None:
    HalfDescribed = Choice[("billing", ""), ("technical", "Bugs")]

    class Pick(dspy.Signature):
        """Route a support ticket."""

        ticket: str = dspy.InputField()
        team: HalfDescribed = dspy.OutputField(desc="Which team?")

    fake.json(body_with(team=choice_answer({"billing": 0.7, "technical": 0.3})))
    predict = dspy.Predict(Pick)
    predict.set_lm(fake.wity())
    assert predict(ticket=TICKET).team.value == "billing"
    assert fake.bodies[0]["questions"]["team"]["criteria"] == {"billing": "billing", "technical": "Bugs"}


def test_noul_with_one_side_is_an_error_before_sending(fake: FakeWity) -> None:
    predict = dspy.Predict(UrgentOnly)
    predict.set_lm(fake.wity())
    predict.set_criteria("urgent", {"true": "Needs action today"})
    with pytest.raises(ValueError, match="both sides of 'urgent'"):
        predict(ticket=TICKET)
    assert fake.requests == []


def test_text_outputs_are_dspys_error(fake: FakeWity) -> None:
    class Mixed(dspy.Signature):
        """Route a support ticket."""

        ticket: str = dspy.InputField()
        urgent: Urgent = dspy.OutputField(desc="Is it urgent?")
        summary: str = dspy.OutputField()

    predict = dspy.Predict(Mixed)
    predict.set_lm(fake.wity())
    with pytest.raises(ValueError, match="summary"):
        predict(ticket=TICKET)
    assert fake.requests == []


# --- errors ----------------------------------------------------------------


def test_sdk_errors_come_through_unchanged(fake: FakeWity) -> None:
    saved = json.loads((Path(__file__).parent / "fixtures" / "error-object-instructions.json").read_text())
    fake.json(saved["body"], status=saved["status"])
    with pytest.raises(wity.BadRequestError, match="instructions must be a nonempty string"):
        triage(fake.wity())(ticket=TICKET)


def test_bad_key_is_authentication_error(fake: FakeWity) -> None:
    saved = json.loads((Path(__file__).parent / "fixtures" / "error-bad-key.json").read_text())
    fake.json(saved["body"], status=saved["status"])
    with pytest.raises(wity.AuthenticationError):
        triage(fake.wity())(ticket=TICKET)


def test_missing_key_fails_when_made() -> None:
    with pytest.raises(wity.WityError, match="No API key found"):
        Wity()


def test_retries_are_the_sdks(fake: FakeWity) -> None:
    # The SDK waits about half a second before its first retry.
    fake.json({"error": "busy"}, status=503)
    assert triage(fake.wity(max_retries=1))(ticket=TICKET).team.value == "billing"
    assert len(fake.requests) == 2


# --- cache, history, usage, callbacks --------------------------------------


def test_cache_hit_skips_the_network(fake: FakeWity) -> None:
    predict = triage(fake.wity(cache=True))
    first = predict(ticket=TICKET)
    second = predict(ticket=TICKET)
    assert len(fake.requests) == 1
    assert second.team.probabilities == first.team.probabilities


async def test_sync_and_async_share_the_cache(fake: FakeWity) -> None:
    predict = triage(fake.wity(cache=True))
    predict(ticket=TICKET)
    await predict.acall(ticket=TICKET)
    assert len(fake.requests) == 1


def test_cache_misses_when_options_change(fake: FakeWity) -> None:
    triage(fake.wity(cache=True))(ticket=TICKET)
    triage(fake.wity(cache=True, reasoning="off"))(ticket=TICKET)
    triage(fake.wity(cache=True, max_latency_ms=500))(ticket=TICKET)
    triage(fake.wity(cache=True, base_url="http://localhost:9999"))(ticket=TICKET)
    triage(fake.wity(cache=True))(ticket="Something else")
    assert len(fake.requests) == 5


def test_cache_off_always_sends(fake: FakeWity) -> None:
    predict = triage(fake.wity(cache=False))
    predict(ticket=TICKET)
    predict(ticket=TICKET)
    assert len(fake.requests) == 2


def test_history(fake: FakeWity) -> None:
    lm = fake.wity()
    triage(lm)(ticket=TICKET)
    assert len(lm.history) == 1
    entry = lm.history[0]
    assert entry["model"] == "wity-1"
    assert entry["cache_hit"] is False
    assert entry["request"]["state"]["inputs"] == {"ticket": TICKET}
    # The full answer is kept, reasoning details included.
    assert entry["response"]["answers"]["team"]["reasoning"]["mode"] == "auto"
    assert json.loads(entry["outputs"][0])["urgent"] == {
        "noul": fixture_body("multi-auto")["answers"]["urgent"]["noul"]
    }


def test_inspect_history_prints(fake: FakeWity, capsys: pytest.CaptureFixture[str]) -> None:
    lm = fake.wity()
    triage(lm)(ticket=TICKET)
    lm.inspect_history()
    printed = capsys.readouterr().out
    assert TICKET in printed
    assert "billing" in printed


def test_usage_tracking(fake: FakeWity) -> None:
    with dspy.context(track_usage=True):
        res = triage(fake.wity())(ticket=TICKET)
    assert res.get_lm_usage() == {"wity-1": {"prompt_tokens": 287, "completion_tokens": 0}}


def test_cache_hits_add_no_usage(fake: FakeWity) -> None:
    predict = triage(fake.wity(cache=True))
    predict(ticket=TICKET)
    with dspy.context(track_usage=True):
        res = predict(ticket=TICKET)
    assert res.get_lm_usage() == {}


def test_callbacks_see_the_call(fake: FakeWity) -> None:
    seen: list[str] = []

    class Spy(BaseCallback):
        def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
            seen.append(f"start {sorted(inputs)}")

        def on_lm_end(self, call_id: str, outputs: Any, exception: Exception | None = None) -> None:
            seen.append(f"end {sorted(outputs)}")

    triage(fake.wity(callbacks=[Spy()]))(ticket=TICKET)
    assert seen == ["start ['questions', 'state']", "end ['severity', 'team', 'urgent']"]


# --- copy, save, load ------------------------------------------------------


def test_copy_keeps_settings_not_history(fake: FakeWity) -> None:
    lm = fake.wity(reasoning="off", max_latency_ms=900)
    triage(lm)(ticket=TICKET)
    other = lm.copy(reasoning="always")
    assert other.history == []
    assert other.reasoning == "always"
    assert other.max_latency_ms == 900
    triage(other)(ticket=TICKET)
    assert fake.bodies[1]["reasoning"] == "always"
    assert fake.requests[1].headers["authorization"] == f"Bearer {TEST_KEY}"


def test_best_of_n_gets_a_clear_error(fake: FakeWity) -> None:
    predict = dspy.Predict(UrgentOnly)
    predict.set_lm(fake.wity())
    best = dspy.BestOfN(predict, N=3, reward_fn=lambda args, pred: 1.0, threshold=1.0)
    with pytest.raises(TypeError, match=r"doesn't sample.*dspy\.BestOfN"):
        best(ticket=TICKET)
    assert fake.requests == []


async def test_generation_settings_error_names_wity(fake: FakeWity) -> None:
    predict = dspy.Predict(UrgentOnly)
    predict.set_lm(fake.wity())
    with pytest.raises(ValueError, match=r"^Unsupported Wity generation settings: \['temperature'\]"):
        predict(ticket=TICKET, config={"temperature": 0.5})
    with pytest.raises(ValueError, match="Unsupported Wity generation settings"):
        await predict.acall(ticket=TICKET, config={"temperature": 0.5})
    assert fake.requests == []


def test_optimizers_that_reset_predictors_work_with_a_context(fake: FakeWity) -> None:
    # LabeledFewShot starts from a fresh copy that drops set_lm(). That's DSPy's behaviour for any LM.
    trainset = [dspy.Example(ticket="a", urgent=True).with_inputs("ticket")]
    with dspy.context(lm=fake.wity()):
        tuned = dspy.LabeledFewShot(k=1).compile(dspy.Predict(UrgentOnly), trainset=trainset)
        tuned(ticket=TICKET)
    assert fake.bodies[0]["state"]["demos"] == [{"ticket": "a", "urgent": True}]


def test_program_deepcopy_shares_the_client_quietly(fake: FakeWity, caplog: pytest.LogCaptureFixture) -> None:
    predict = triage(fake.wity())
    with caplog.at_level(logging.WARNING):
        other = predict.deepcopy()
    assert other.lm is predict.lm
    assert caplog.text == ""


def test_dump_state_has_no_key(fake: FakeWity) -> None:
    state = fake.wity(reasoning="off").dump_state()
    assert state == {
        "_dspy_lm_class": "dspy_wity.client.Wity",
        "reasoning": "off",
        "max_latency_ms": None,
        "base_url": "https://api.wity.alphanimble.com",
        "timeout": 60.0,
        "max_retries": 0,
        "cache": False,
    }


def test_saved_program_loads_with_wity(fake: FakeWity, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    predict = triage(fake.wity(reasoning="off"))
    predict.fields["urgent"] = {"threshold": 0.8}
    path = tmp_path / "triage.json"
    predict.save(str(path))
    assert TEST_KEY not in path.read_text()

    monkeypatch.setenv("WITY_API_KEY", TEST_KEY)
    loaded = dspy.Predict(Triage)
    loaded.load(str(path), allow_unsafe_lm_state=True)
    assert isinstance(loaded.lm, Wity)
    assert loaded.lm.reasoning == "off"
    assert loaded.fields == {"urgent": {"threshold": 0.8}}


def test_saved_program_needs_dspys_trust_flag(fake: FakeWity, tmp_path: Path) -> None:
    path = tmp_path / "triage.json"
    triage(fake.wity()).save(str(path))
    with pytest.raises(ValueError, match="allow_unsafe_lm_state=True"):
        dspy.Predict(Triage).load(str(path))


def test_saved_program_without_an_lm_loads_anywhere(fake: FakeWity, tmp_path: Path) -> None:
    predict = triage(fake.wity())
    predict.fields["urgent"] = {"threshold": 0.8}
    predict.set_lm(None)
    path = tmp_path / "triage.json"
    predict.save(str(path))
    assert json.loads(path.read_text())["lm"] is None
    loaded = dspy.Predict(Triage)
    loaded.load(str(path))
    loaded.set_lm(fake.wity())
    assert loaded.fields == {"urgent": {"threshold": 0.8}}
    assert loaded(ticket=TICKET).team.value == "billing"


def test_pickle_keeps_settings_not_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    lm = Wity(api_key=TEST_KEY, reasoning="off", max_latency_ms=700, max_retries=1, cache=False)
    data = pickle.dumps(lm)
    assert TEST_KEY.encode() not in data
    with pytest.raises(wity.WityError, match="No API key found"):
        pickle.loads(data)
    monkeypatch.setenv("WITY_API_KEY", "wity_other_process_key")
    other = pickle.loads(data)
    assert (other.reasoning, other.max_latency_ms, other.max_retries, other.cache) == ("off", 700, 1, False)
    assert other.history == []


def test_whole_program_save_and_load(fake: FakeWity, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WITY_API_KEY", TEST_KEY)
    predict = dspy.Predict(Triage)
    predict.set_lm(Wity(reasoning="off", base_url="http://localhost:9999"))
    predict.save(str(tmp_path / "program"), save_program=True)
    assert TEST_KEY.encode() not in b"".join(f.read_bytes() for f in (tmp_path / "program").rglob("*") if f.is_file())
    loaded = dspy.load(str(tmp_path / "program"), allow_pickle=True)
    assert isinstance(loaded.lm, Wity)
    assert (loaded.lm.reasoning, loaded.lm.base_url) == ("off", "http://localhost:9999")


def test_global_lm(fake: FakeWity) -> None:
    with dspy.context(lm=fake.wity()):
        assert dspy.Predict(Triage)(ticket=TICKET).team.value == "billing"


# --- the key never leaks ---------------------------------------------------


def test_key_never_leaks(fake: FakeWity, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WITY_LOG_LEVEL", "debug")
    lm = fake.wity()
    with caplog.at_level(logging.DEBUG):
        triage(lm)(ticket=TICKET)
    assert TEST_KEY not in repr(lm)
    assert TEST_KEY not in json.dumps(lm.history, default=str)
    assert TEST_KEY not in json.dumps(lm.dump_state())
    assert TEST_KEY not in caplog.text


# --- ReAnchor --------------------------------------------------------------


def test_reanchor_tunes_the_threshold(fake: FakeWity) -> None:
    # Wity's yes-probability for each ticket: 0.05, 0.10 ... 0.95. The right answer is yes from 0.75 up,
    # so the default threshold of 0.5 gets five tickets wrong.
    probabilities = {f"ticket {i}": i / 20 for i in range(1, 20)}
    labels = {ticket: p >= 0.75 for ticket, p in probabilities.items()}

    def answer(request: httpx.Request) -> httpx.Response:
        ticket = json.loads(request.content)["state"]["inputs"]["ticket"]
        return httpx.Response(200, json=only_urgent(probabilities[ticket]))

    fake.default = answer
    predict = dspy.Predict(UrgentOnly)
    predict.set_lm(fake.wity(cache=True))
    trainset = [dspy.Example(ticket=t, urgent=labels[t]).with_inputs("ticket") for t in probabilities]

    def metric(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
        return float(pred.urgent.value == example.urgent)

    tuned = dspy.experimental.ReAnchor(metric, num_threads=1).compile(predict, trainset=trainset)
    threshold = tuned.fields["urgent"]["threshold"]
    assert 0.7 < threshold <= 0.75
    assert predict.fields == {}
    # One request per ticket. Every other setting it tried reused the cached answers.
    assert len(fake.requests) == len(trainset)
    assert all(tuned(ticket=t).urgent.value == labels[t] for t in probabilities)


def test_reanchor_needs_the_cache(fake: FakeWity) -> None:
    predict = dspy.Predict(UrgentOnly)
    predict.set_lm(fake.wity(cache=False))
    trainset = [dspy.Example(ticket="a", urgent=True).with_inputs("ticket")]
    with pytest.raises(ValueError, match="do not cache"):
        dspy.experimental.ReAnchor(lambda *a: 1.0).compile(predict, trainset=trainset)


def test_score_cuts(fake: FakeWity) -> None:
    class Rate(dspy.Signature):
        """Check a support ticket."""

        ticket: str = dspy.InputField()
        severity: Severity = dspy.OutputField(desc="How bad?")

    fake.default = httpx.Response(200, json=body_with(severity=score_answer([0.2, 0.5, 0.3])))
    predict = dspy.Predict(Rate)
    predict.set_lm(fake.wity())
    assert predict(ticket=TICKET).severity.level == 1
    predict.fields["severity"] = {"cuts": [0.5, 1.05]}
    assert predict(ticket=TICKET).severity.level == 2
