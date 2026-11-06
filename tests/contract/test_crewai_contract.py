"""CrewAI 1.15.36 behaviour the mock adapter depends on (§ARCH 2.2, pitfalls P1, P3, P4, P5)."""

from __future__ import annotations

import inspect
import json
import logging
import subprocess
import sys
from typing import Any

import pytest
from crewai import Agent, Crew
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.i18n import I18N

from tests.contract import crewai_recorder as rec

TEMPLATE_OBSERVATION = "Observation: the result of the action"


@pytest.fixture(scope="module")
def single_tool() -> rec.Scenario:
    return rec.run_single_tool()


@pytest.fixture(scope="module")
def handoff() -> rec.Scenario:
    return rec.run_context_handoff()


def _calls(s: rec.Scenario, key: str = "agent") -> list[rec.RecordedCall]:
    return s.llms[key].recorded


# ---------------------------------------------------------------- BaseLLM surface (§2.2.1)


def test_base_llm_has_exactly_one_abstract_method() -> None:
    assert BaseLLM.__abstractmethods__ == frozenset({"call"})


def test_call_signature_has_the_parameters_the_adapter_uses() -> None:
    params = inspect.signature(BaseLLM.call).parameters
    assert list(params) == [
        "self", "messages", "tools", "callbacks", "available_functions",
        "from_task", "from_agent", "response_model",
    ]  # fmt: skip


def test_base_llm_does_not_define_supports_function_calling() -> None:
    """§2.2.1 amended: CrewAI picks native tool calling only if the LLM *has* a callable
    `supports_function_calling()` returning True, but its converter calls the method
    unconditionally — so the adapter must define it (returning False) itself."""
    assert not hasattr(BaseLLM, "supports_function_calling")
    llm = rec.RecordingLLM()
    assert llm.model == "mock-llm"
    assert llm.supports_function_calling() is False
    assert llm.supports_stop_words() is True


# ---------------------------------------------------------------- defaults we override (§2.2.3–5)


def test_agent_defaults_justify_our_explicit_overrides() -> None:
    """P4: Agent.cache defaults to True, which would hide repeated tool calls."""
    fields = Agent.model_fields
    assert fields["cache"].default is True
    assert fields["allow_delegation"].default is False
    assert fields["max_iter"].default == 25
    assert fields["max_retry_limit"].default == 2
    agent = Agent(role="r", goal="g", backstory="b", llm=rec.RecordingLLM())
    assert agent.cache is True


def test_crew_defaults() -> None:
    fields = Crew.model_fields
    assert (fields["cache"].default, fields["memory"].default, fields["planning"].default) == (
        False,
        False,
        False,
    )


def test_recorder_agents_carry_the_overrides(single_tool: rec.Scenario) -> None:
    agent = _calls(single_tool)[0]
    assert agent.from_agent == "Agent"
    built = rec._agent(rec.RecordingLLM(), "x", [])
    assert (built.cache, built.allow_delegation, built.max_iter) == (False, False, 4)


# ---------------------------------------------------------------- ReAct message layout (§2.2.2)


def test_one_tool_round_trip_takes_exactly_two_llm_calls(single_tool: rec.Scenario) -> None:
    """(d) no hidden converter re-ask: Action call + Final Answer call."""
    assert len(_calls(single_tool)) == 2


def test_result_pydantic_is_filled_from_the_final_answer(single_tool: rec.Scenario) -> None:
    assert single_tool.result.pydantic == rec.Answer(answer="ok", n=2)
    assert json.loads(single_tool.result.raw) == {"answer": "ok", "n": 2}


def test_first_call_layout_is_system_then_current_task(single_tool: rec.Scenario) -> None:
    first = _calls(single_tool)[0].messages
    assert [m["role"] for m in first] == ["system", "user"]
    assert "Current Task:" in first[1]["content"]
    assert "<<<QUERY>>>hello<<<END_QUERY>>>" in first[1]["content"]


def test_system_template_contains_the_literal_observation_line(single_tool: rec.Scenario) -> None:
    """(a) P1: the template line is never a real observation."""
    system = _calls(single_tool)[0].messages[0]["content"]
    assert TEMPLATE_OBSERVATION in system
    assert rec.MARKER not in system


def test_observation_is_appended_inside_our_own_assistant_message(
    single_tool: rec.Scenario,
) -> None:
    """(b) the tool result lives in OUR marked assistant message, up to its end."""
    second = _calls(single_tool)[1].messages
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    ours = second[2]["content"]
    assert rec.MARKER in ours
    head, sep, observation = ours.partition("\nObservation:")
    assert sep
    assert "Action: echo_tool" in head
    assert json.loads(observation) == {"kind": "echo_result", "echo": "hello"}


def test_next_user_message_is_the_post_tool_reasoning_nudge(single_tool: rec.Scenario) -> None:
    """(c) P3: the nudge follows our message; a naive 'after last Observation:' parse eats it."""
    nudge = I18N().slice("post_tool_reasoning")
    assert nudge.startswith("Analyze the tool result.")
    assert _calls(single_tool)[1].messages[3]["content"] == nudge


def test_response_model_is_not_passed_to_a_tool_using_agent(single_tool: rec.Scenario) -> None:
    assert [c.response_model for c in _calls(single_tool)] == [None, None]
    assert [c.tools_passed for c in _calls(single_tool)] == [False, False]
    assert {c.from_task for c in _calls(single_tool)} == {"Task"}


# ---------------------------------------------------------------- task context hand-off


def test_context_output_reaches_the_next_task_as_raw_text(handoff: rec.Scenario) -> None:
    """The Composer finds prior task outputs by their JSON `kind` (§ARCH 7.8.1 step 4)."""
    assert [len(handoff.llms["first"].recorded), len(handoff.llms["second"].recorded)] == [2, 1]
    user = handoff.llms["second"].recorded[0].messages[1]["content"]
    _, sep, context = user.partition("This is the context you're working with:\n")
    assert sep
    assert context.startswith('{"kind": "echo_result", "echo": "hello"}')
    assert handoff.result.pydantic == rec.Answer(answer="ok", n=2)


def test_tool_less_pydantic_agent_receives_the_response_model(handoff: rec.Scenario) -> None:
    """The Composer (no tools, output_pydantic) gets response_model=<Model> on its call."""
    assert [c.response_model for c in handoff.llms["second"].recorded] == ["Answer"]
    assert [c.response_model for c in handoff.llms["first"].recorded] == [None, None]


def test_invalid_final_answer_triggers_three_converter_re_asks() -> None:
    """A non-schema Final Answer costs 1 + 3 LLM calls (Converter.max_attempts), each
    [system=format instructions, user=raw output]; then pydantic is None and raw is kept.
    Phase 6 asserts exact call counts, so a mock emitting bad JSON is caught; the
    orchestrator.s `CrewResponse.model_validate_json(result.raw)` stays the real gate."""
    scenario = rec.run_invalid_final_answer()
    calls = _calls(scenario)
    assert len(calls) == 4
    for call in calls[1:]:
        assert [m["role"] for m in call.messages] == ["system", "user"]
        assert call.messages[0]["content"].startswith("Format your final answer according to")
        assert call.messages[1]["content"] == "not json at all"
        assert call.response_model is None
    assert scenario.result.pydantic is None
    assert scenario.result.raw == "not json at all"


# ---------------------------------------------------------------- fixtures replay reality


@pytest.mark.parametrize("name", sorted(rec.SCENARIOS))
def test_committed_fixture_equals_a_fresh_recording(name: str) -> None:
    """Phase 6 unit tests replay these files; regenerate with `make capture-crewai-fixtures`."""
    path = rec.FIXTURE_DIR / f"{name}.json"
    committed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    assert committed == rec.recording(name)


# ---------------------------------------------------------------- no TTY, no stdin, no network


@pytest.mark.timeout(20)
def test_kickoff_completes_with_stdin_closed_and_no_tty() -> None:
    """P5: an interactive 'first-time trace' prompt would hang here."""
    code = (
        "from tests.contract import crewai_recorder as r; "
        "s = r.run_single_tool(); print('PYDANTIC', s.result.pydantic)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=18,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "PYDANTIC answer='ok' n=2" in proc.stdout


@pytest.mark.parametrize(
    ("messages", "tokens"),
    [
        ("abcd" * 5, 5),
        ([{"role": "user", "content": "x" * 10}], 3),
        # summed over messages, then rounded up once: ceil((3 + 5) / 4) == 2
        ([{"role": "system", "content": "abc"}, {"role": "user", "content": "defgh"}], 2),
    ],
)
def test_custom_llm_token_estimate_is_ceil_total_chars_over_four(
    caplog: pytest.LogCaptureFixture, messages: Any, tokens: int
) -> None:
    """§2.2.8: our budget estimator uses the same ceil(total_chars/4) heuristic."""
    with caplog.at_level(logging.WARNING):
        assert rec.RecordingLLM().get_input_tokens_count(messages) == tokens
    assert "one token per four characters" in caplog.text
