"""Explicit interview workflow with buffered answer output."""

from collections.abc import Awaitable, Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from backend.agents.final_judge import judge_answer
from backend.agents.guard import evaluate_query
from backend.config import Settings, get_settings
from backend.policy import (
    FALLBACK_NO_INFORMATION,
    FALLBACK_OFF_TOPIC,
    FALLBACK_UNRELIABLE,
)
from backend.schemas import Evidence, GuardResult, JudgeResult


class GraphState(TypedDict, total=False):
    query: str
    main_history: list[dict[str, str]]
    guard_relevant: bool
    guard_greeting: bool
    evidence: list[Evidence]
    answer: str
    response: str
    approved: bool


AnswerGenerator = Callable[[str, list[Evidence], list[dict[str, str]]], Awaitable[str]]
EvidenceRetriever = Callable[[str], Awaitable[list[Evidence]]]
GuardEvaluator = Callable[[str], Awaitable[GuardResult]]
AnswerJudge = Callable[[str, str, str], Awaitable[JudgeResult]]


def build_graph(
    *,
    settings: Settings | None = None,
    retrieve: EvidenceRetriever | None = None,
    generate: AnswerGenerator | None = None,
    guard: GuardEvaluator | None = None,
    judge: AnswerJudge | None = None,
):
    runtime_settings = settings or get_settings()

    async def guard_node(state: GraphState) -> GraphState:
        result = await (
            guard(state["query"])
            if guard is not None
            else evaluate_query(state["query"], runtime_settings)
        )
        return {
            "guard_relevant": result.is_relevant,
            "guard_greeting": result.is_greeting,
            "response": (
                result.reason if result.is_greeting else None
            ) if result.is_relevant else FALLBACK_OFF_TOPIC,
        }

    async def retrieve_node(state: GraphState) -> GraphState:
        if retrieve is None:
            return {"evidence": []}
        return {"evidence": await retrieve(state["query"])}

    async def answer_node(state: GraphState) -> GraphState:
        evidence = state.get("evidence", [])
        if not evidence:
            return {"response": FALLBACK_NO_INFORMATION}
        if generate is None:
            return {"response": FALLBACK_UNRELIABLE}
        answer = await generate(
            state["query"], evidence, state.get("main_history", [])
        )
        return {"answer": answer}

    async def judge_node(state: GraphState) -> GraphState:
        if "answer" not in state:
            return {}
        evidence_text = "\n".join(item.text for item in state.get("evidence", []))
        result = await (
            judge(state["query"], state["answer"], evidence_text)
            if judge is not None
            else judge_answer(
                state["query"], state["answer"], evidence_text, runtime_settings
            )
        )
        return {
            "approved": result.approved,
            "response": state["answer"] if result.approved else FALLBACK_UNRELIABLE,
        }

    def after_guard(state: GraphState) -> str:
        if state.get("guard_greeting"):
            return "finish"
        return "retrieve" if state.get("guard_relevant") else "finish"

    def after_answer(state: GraphState) -> str:
        return "judge" if "answer" in state else "finish"

    workflow = StateGraph(GraphState)
    workflow.add_node("guard", guard_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("judge", judge_node)
    workflow.add_edge(START, "guard")
    workflow.add_conditional_edges("guard", after_guard, {"retrieve": "retrieve", "finish": END})
    workflow.add_edge("retrieve", "answer")
    workflow.add_conditional_edges("answer", after_answer, {"judge": "judge", "finish": END})
    workflow.add_edge("judge", END)
    return workflow.compile()