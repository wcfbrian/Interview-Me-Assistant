"""Top-level interview graph.

The graph owns only request-level routing. Resume and project retrieval are
tools used by ``main_agent`` and are intentionally not graph nodes.
"""

from __future__ import annotations

from contextvars import ContextVar
import json
import operator
from typing import Annotated, Any, Literal, TypedDict
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph

from src.backend.artifacts import load_profile
from src.backend.config import settings
from src.backend.config import tracing_config
from src.backend.graph.decisions import classify_with_model
from src.backend.graph.main_agent import build_main_agent

GuardRoute = Literal["safe", "greeting", "prohibited"]

CAPTURED_TOOL_NAMES = {
    "get_name",
    "get_self_introduction",
    "get_working_background",
    "get_education",
    "get_skills",
    "get_highlighted_project",
    "get_project_detail",
}
_tool_responses: ContextVar[list[str] | None] = ContextVar("tool_responses", default=None)


def _tool_response_text(output: Any, tool_name: str) -> list[str]:
    content = getattr(output, "content", output)
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return [content]
        if tool_name == "get_project_detail":
            evidence = payload.get("evidence", []) if isinstance(payload, dict) else []
            return [
                item["text"]
                for item in evidence
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
        if isinstance(payload, dict) and isinstance(payload.get("text"), str):
            return [payload["text"]]
    return [str(content)]


class GraphState(MessagesState):
    """Conversation state plus evidence returned by approved retrieval tools."""

    context: Annotated[list[str], operator.add]


class _ToolResponseCollector(BaseCallbackHandler):
    """Collect outputs only from tools whose results form interview evidence."""

    def __init__(self) -> None:
        self._tool_names: dict[UUID, str] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        _: str,
        *,
        run_id: UUID,
        **__: Any,
    ) -> None:
        tool_name = serialized.get("name")
        if tool_name in CAPTURED_TOOL_NAMES:
            self._tool_names[run_id] = tool_name

    def on_tool_end(self, output: Any, *, run_id: UUID, **__: Any) -> None:
        responses = _tool_responses.get()
        tool_name = self._tool_names.get(run_id)
        if tool_name is not None and responses is not None:
            responses.extend(_tool_response_text(output, tool_name))


def _guard_context(messages: list[BaseMessage]) -> str:
    """Format the conversation so the guard can classify follow-up questions."""
    if len(messages) == 1:
        return str(messages[0].content)
    turns = []
    for message in messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        turns.append(f"{role}: {message.content}")
    return "\n".join(turns)


def guard_route(state: GraphState) -> GuardRoute:
    """Route the request using the mandatory structured guard model."""
    context = _guard_context(state["messages"]).strip()
    category = classify_with_model(context).category
    return category


def greeting(state: GraphState) -> dict[str, list[BaseMessage]]:
    """Return a concise response for common greeting patterns."""
    latest_message = state["messages"][-1] if state["messages"] else None

    greeting = str(latest_message.content).strip().lower() if latest_message else ""

    if greeting.rstrip("?!., ") in {"how are you", "how are you doing", "how is it going"}:
        response = "I am fine. Thank you for asking."
    else:
        response = "Hello. Welcome to interview me."

    return {"messages": [AIMessage(content=response)]}


def refuse_to_answer(_: GraphState) -> dict[str, list[BaseMessage]]:
    return {"messages": [AIMessage(content="I am sorry. I am not sure. Please ask another question.")]}


async def main_agent(state: GraphState) -> dict[str, list[BaseMessage] | list[str]]:
    """Stream the supervisor's final answer while preserving the graph message state."""
    writer = get_stream_writer()
    answer_parts: list[str] = []
    tool_responses: list[str] = []
    context_token = _tool_responses.set(tool_responses)
    collector = _ToolResponseCollector()
    config = tracing_config("interview-main-agent", "main-agent")
    config["callbacks"] = [collector]

    try:
        async for chunk, _ in build_main_agent().astream(
            {"messages": state["messages"]},
            context=load_profile(settings.profile_path),
            config=config,
            stream_mode="messages",
        ):
            if not isinstance(chunk, AIMessageChunk) or not isinstance(chunk.content, str):
                continue
            if chunk.content:
                answer_parts.append(chunk.content)
                writer(chunk.content)
    finally:
        _tool_responses.reset(context_token)

    return {
        "messages": [AIMessage(content="".join(answer_parts))],
        "context": tool_responses,
    }


def build_graph():
    """Compile the request-level graph with conversation and evidence state."""
    builder = StateGraph(GraphState)
    builder.add_node("main_agent", main_agent)
    builder.add_node("greeting", greeting)
    builder.add_node("refuse_to_answer", refuse_to_answer)
    builder.add_conditional_edges(
        START,
        guard_route,
        {
            "safe": "main_agent",
            "greeting": "greeting",
            "prohibited": "refuse_to_answer",
        },
    )
    builder.add_edge("main_agent", END)
    builder.add_edge("greeting", END)
    builder.add_edge("refuse_to_answer", END)
    return builder.compile()

