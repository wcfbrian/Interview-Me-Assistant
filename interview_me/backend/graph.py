"""Top-level interview graph.

The graph owns only request-level routing. Resume and project retrieval are
tools used by ``main_agent`` and are intentionally not graph nodes.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph

from interview_me.backend.agents.main import _build_main_agent
from interview_me.backend.config import tracing_config
from interview_me.backend.decisions import classify_with_model

GuardRoute = Literal["safe", "greeting", "prohibited"]


def _guard_context(messages: list[BaseMessage]) -> str:
    """Format the conversation so the guard can classify follow-up questions."""
    if len(messages) == 1:
        return str(messages[0].content)
    turns = []
    for message in messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        turns.append(f"{role}: {message.content}")
    return "\n".join(turns)


def guard_route(state: MessagesState) -> GuardRoute:
    """Route the request using the mandatory structured guard model."""
    context = _guard_context(state["messages"]).strip()
    category = classify_with_model(context).category
    return category
    


def fixed_greeting(_: MessagesState) -> dict[str, list[BaseMessage]]:
    return {"messages": [AIMessage(content="Hello. Welcome to interview me.")]}


def fixed_relevance_response(_: MessagesState) -> dict[str, list[BaseMessage]]:
    return {"messages": [AIMessage(content="Please ask relevant questions to interview me.")]}


async def main_agent(state: MessagesState) -> dict[str, list[BaseMessage]]:
    """Stream the supervisor's final answer while preserving the graph message state."""
    writer = get_stream_writer()
    answer_parts: list[str] = []

    async for chunk, _ in _build_main_agent().astream(
        {"messages": state["messages"]},
        config=tracing_config("interview-main-agent", "main-agent"),
        stream_mode="messages",
    ):
        if not isinstance(chunk, AIMessageChunk) or not isinstance(chunk.content, str):
            continue
        if chunk.content:
            answer_parts.append(chunk.content)
            writer(chunk.content)

    return {"messages": [AIMessage(content="".join(answer_parts))]}


def build_graph():
    """Compile the request-level graph with messages as its only state."""
    builder = StateGraph(MessagesState)
    builder.add_node("main_agent", main_agent)
    builder.add_node("fixed_greeting", fixed_greeting)
    builder.add_node("fixed_relevance_response", fixed_relevance_response)
    builder.add_conditional_edges(
        START,
        guard_route,
        {
            "safe": "main_agent",
            "greeting": "fixed_greeting",
            "prohibited": "fixed_relevance_response",
        },
    )
    builder.add_edge("main_agent", END)
    builder.add_edge("fixed_greeting", END)
    builder.add_edge("fixed_relevance_response", END)
    return builder.compile()


graph = build_graph()