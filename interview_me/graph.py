"""Top-level interview graph.

The graph owns only request-level routing. Resume and project retrieval are
tools used by ``main_agent`` and are intentionally not graph nodes.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from interview_me.backend.agents.main import run_main_agent
from interview_me.backend.decisions import classify_with_model

GuardRoute = Literal["relevant", "greeting", "irrelevant"]


def _last_user_message(messages: list[BaseMessage]) -> str:
    """Return the latest user query, or an empty string for malformed input."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def guard_route(state: MessagesState) -> GuardRoute:
    """Route the request using the mandatory structured guard model."""
    query = _last_user_message(state["messages"]).strip().lower()
    category = classify_with_model(query).category
    return "irrelevant" if category == "prohibited" else category
    


def fixed_greeting(_: MessagesState) -> dict[str, list[BaseMessage]]:
    return {"messages": [AIMessage(content="Hello. Please ask me about my background or projects.")]}


def fixed_relevance_response(_: MessagesState) -> dict[str, list[BaseMessage]]:
    return {"messages": [AIMessage(content="Please ask relevant questions to interview me.")]}


def main_agent(state: MessagesState) -> dict[str, list[BaseMessage]]:
    """Call the supervisor; its tools return evidence into the same message loop."""
    return {"messages": [run_main_agent(state["messages"])]}


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
            "relevant": "main_agent",
            "greeting": "fixed_greeting",
            "irrelevant": "fixed_relevance_response",
        },
    )
    builder.add_edge("main_agent", END)
    builder.add_edge("fixed_greeting", END)
    builder.add_edge("fixed_relevance_response", END)
    return builder.compile()


graph = build_graph()