"""Single-pass adapter from curated cases to LangGraph responses."""

from __future__ import annotations

from dataclasses import dataclass
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from interview_me.backend.evaluation.cases import EvaluationCase, GuardRoute
from interview_me.backend.graph import graph


@dataclass(frozen=True)
class EvaluationResult:
    case: EvaluationCase
    answer: str
    route: GuardRoute
    latency_ms: float


def _messages(case: EvaluationCase) -> list[BaseMessage]:
    return [
        HumanMessage(content=turn.content)
        if turn.role == "user"
        else AIMessage(content=turn.content)
        for turn in case.conversation
    ]


async def run_case(case: EvaluationCase) -> EvaluationResult:
    """Run a case once and infer its route from LangGraph's streamed node update."""
    started = time.perf_counter()
    route: GuardRoute | None = None
    answer = ""

    async for update in graph.astream(
        {"messages": _messages(case)}, stream_mode="updates"
    ):
        for node_name, value in update.items():
            if node_name == "main_agent":
                route = "safe"
            elif node_name == "fixed_greeting":
                route = "greeting"
            elif node_name == "fixed_relevance_response":
                route = "prohibited"

            messages = value.get("messages", []) if isinstance(value, dict) else []
            for message in messages:
                if isinstance(message, AIMessage) and isinstance(message.content, str):
                    answer = message.content

    if route is None:
        raise RuntimeError(f"Graph did not emit a route for evaluation case {case.case_id}")

    return EvaluationResult(
        case=case,
        answer=answer,
        route=route,
        latency_ms=(time.perf_counter() - started) * 1000,
    )