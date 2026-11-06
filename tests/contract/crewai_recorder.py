"""Record the real message lists CrewAI hands to a custom LLM (§ARCH 2.2).

Used two ways:
* by tests/contract/test_crewai_contract.py, which asserts the live layout, and that the
  committed fixtures still equal a fresh recording (so Phase 6 unit tests replay reality);
* by `make capture-crewai-fixtures`, which (re)writes tests/fixtures/crewai_messages/*.json.

The scripted LLM is deliberately dumb and generic: it calls its single tool once (marking its
own message), then returns a fixed JSON Final Answer. No library internals are patched.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import credagent  # noqa: F401  (egress-off defaults before crewai is imported)

MARKER = "⟦contract-action:1⟧"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "crewai_messages"


class Answer(BaseModel):
    answer: str
    n: int


class EchoArgs(BaseModel):
    query: str = Field(..., min_length=1)


class EchoTool(BaseTool):
    name: str = "echo_tool"
    description: str = "Echo the query back as JSON."
    args_schema: type[BaseModel] = EchoArgs

    def _run(self, query: str) -> str:
        return json.dumps({"kind": "echo_result", "echo": query})


@dataclass
class RecordedCall:
    messages: list[dict[str, Any]]
    response_model: str | None
    tools_passed: bool
    from_agent: str | None
    from_task: str | None


class RecordingLLM(BaseLLM):
    """Scripted ReAct LLM that records every `call()`."""

    final_answer: str = '{"answer": "ok", "n": 2}'
    use_tool: bool = True
    recorded: list[RecordedCall] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model="mock-llm", **kwargs)
        self.recorded = []

    def supports_function_calling(self) -> bool:
        return False

    def call(  # type: ignore[override]  # noqa: PLR0917  (mirrors BaseLLM.call)
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        msgs = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else [dict(m) for m in messages]
        )
        self.recorded.append(
            RecordedCall(
                messages=msgs,
                response_model=None if response_model is None else response_model.__name__,
                tools_passed=tools is not None,
                from_agent=None if from_agent is None else type(from_agent).__name__,
                from_task=None if from_task is None else type(from_task).__name__,
            )
        )
        already_acted = any(m["role"] == "assistant" and MARKER in m["content"] for m in msgs)
        if self.use_tool and not already_acted:
            return (
                f"Thought: {MARKER} I need the tool\n"
                "Action: echo_tool\n"
                'Action Input: {"query": "hello"}'
            )
        return f"Thought: I now know the final answer\nFinal Answer: {self.final_answer}"


def _agent(llm: RecordingLLM, role: str, tools: list[BaseTool]) -> Agent:
    return Agent(
        role=role,
        goal="Answer the task.",
        backstory="A contract-test agent.",
        tools=tools,
        llm=llm,
        cache=False,
        allow_delegation=False,
        max_iter=4,
        max_retry_limit=1,
        verbose=False,
    )


def _crew(agents: list[Agent], tasks: list[Task]) -> Crew:
    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        cache=False,
        memory=False,
        planning=False,
        verbose=False,
    )


@dataclass
class Scenario:
    llms: dict[str, RecordingLLM]
    result: Any
    tool_calls: list[str] = field(default_factory=list)


def run_single_tool() -> Scenario:
    """One agent, one tool, `output_pydantic`: Action → Observation → Final Answer."""
    llm = RecordingLLM()
    agent = _agent(llm, "Echo Specialist", [EchoTool()])
    task = Task(
        description="Echo the query. <<<QUERY>>>hello<<<END_QUERY>>>",
        expected_output="A JSON object.",
        agent=agent,
        output_pydantic=Answer,
    )
    return Scenario({"agent": llm}, _crew([agent], [task]).kickoff())


def run_context_handoff() -> Scenario:
    """Tool-using first task whose output is the `context` of a tool-less pydantic task."""
    first = RecordingLLM()
    first.final_answer = '{"kind": "echo_result", "echo": "hello"}'
    second = RecordingLLM()
    second.use_tool = False
    a1 = _agent(first, "Echo Specialist", [EchoTool()])
    a2 = _agent(second, "Composer", [])
    t1 = Task(
        description="Echo the query. <<<QUERY>>>hello<<<END_QUERY>>>",
        expected_output="A JSON object.",
        agent=a1,
    )
    t2 = Task(
        description="Compose from context.",
        expected_output="A JSON object.",
        agent=a2,
        context=[t1],
        output_pydantic=Answer,
    )
    return Scenario({"first": first, "second": second}, _crew([a1, a2], [t1, t2]).kickoff())


def run_invalid_final_answer() -> Scenario:
    """A Final Answer that is not schema JSON (what happens on the converter path)."""
    llm = RecordingLLM()
    llm.use_tool = False
    llm.final_answer = "not json at all"
    agent = _agent(llm, "Echo Specialist", [])
    task = Task(
        description="Answer.", expected_output="A JSON object.", agent=agent, output_pydantic=Answer
    )
    return Scenario({"agent": llm}, _crew([agent], [task]).kickoff())


SCENARIOS = {
    "single_tool": run_single_tool,
    "context_handoff": run_context_handoff,
}


def recording(name: str) -> dict[str, list[list[dict[str, Any]]]]:
    """Message lists per LLM, per call — the fixture format."""
    scenario = SCENARIOS[name]()
    return {key: [c.messages for c in llm.recorded] for key, llm in scenario.llms.items()}


def main(argv: list[str]) -> int:
    outdir = Path(argv[1]) if len(argv) > 1 else FIXTURE_DIR
    for name in SCENARIOS:
        path = outdir / f"{name}.json"
        path.write_text(
            json.dumps(recording(name), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
