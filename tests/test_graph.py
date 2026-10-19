import asyncio
from importlib import import_module

import pytest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from src.backend.graph.decisions import GuardDecision, classify_with_model
from src.backend.graph import GraphState, build_graph, greeting, guard_route
from src.backend.graph.main_agent import MAIN_AGENT_PROMPT
from src.backend.graph.resume import _profile_field_tool

graph_module = import_module("src.backend.graph.graph")


def _mock_guard(query):
    normalized = query.lower()
    if normalized in {"hi", "hello", "hey"}:
        return GuardDecision(category="greeting")
    if "education" in normalized or "project" in normalized or "name" in normalized:
        return GuardDecision(category="safe")
    return GuardDecision(category="prohibited")


def test_profile_field_tool_formats_all_fields_as_markdown():
    fields = {
        "name": "Name",
        "self_introduction": "Self Introduction",
        "working_background": "Working Background",
        "education": "Education",
        "skills": "Skills",
        "highlighted_project": "Highlighted Project",
    }
    profile = {field: f"{label} evidence\n\n- Detail" for field, label in fields.items()}
    runtime = SimpleNamespace(context=profile)

    for field, label in fields.items():
        assert _profile_field_tool(field, runtime) == (
            f"## {label}\n"
            f"**Source:** `resume:{field}`\n\n"
            f"{profile[field]}"
        )


# Verifies interview questions reach the main route by replacing model classification with a safe decision.
def test_guard_routes_interview_question_to_main_agent(monkeypatch):
    monkeypatch.setattr(graph_module, "classify_with_model", _mock_guard)
    assert guard_route({"messages": [HumanMessage(content="Tell me about your projects")]}) == "safe"


# Verifies follow-ups retain earlier context by capturing the text passed to the mocked guard.
def test_guard_passes_conversation_context_for_followup(monkeypatch):
    captured = {}

    def fake_guard(context):
        captured["context"] = context
        return GuardDecision(category="safe")

    monkeypatch.setattr(graph_module, "classify_with_model", fake_guard)
    route = guard_route({
        "messages": [
            HumanMessage(content="Tell me about your projects"),
            AIMessage(content="I built a grounded chatbot."),
            HumanMessage(content="What about that project?"),
        ]
    })

    assert route == "safe"
    assert "Tell me about your projects" in captured["context"]
    assert "What about that project?" in captured["context"]


# Verifies candidate-name questions are allowed by evaluating the deterministic safe guard substitute.
def test_guard_allows_candidate_name_question(monkeypatch):
    monkeypatch.setattr(graph_module, "classify_with_model", _mock_guard)
    assert guard_route({"messages": [HumanMessage(content="What is your name?")]}) == "safe"


# Verifies greetings bypass the main agent by evaluating the deterministic greeting guard substitute.
def test_guard_routes_greeting_to_fixed_response(monkeypatch):
    monkeypatch.setattr(graph_module, "classify_with_model", _mock_guard)
    assert guard_route({"messages": [HumanMessage(content="hello")]}) == "greeting"


def test_greeting_handles_common_greeting_patterns():
    salutation = greeting({"messages": [HumanMessage(content="Good morning!")]})
    check_in = greeting({"messages": [HumanMessage(content="How are you?")]})

    assert salutation["messages"][0].content == "Hello. Welcome to interview me."
    assert check_in["messages"][0].content == "I am fine. Thank you for asking."


# Verifies the compiled graph exposes only request-level workflow nodes by inspecting its node set.
def test_graph_contains_only_request_level_nodes():
    graph = build_graph()
    assert set(graph.nodes) == {
        "__start__",
        "main_agent",
        "greeting",
        "refuse_to_answer",
    }


def test_graph_invokes_greeting_node(monkeypatch):
    monkeypatch.setattr(graph_module, "classify_with_model", _mock_guard)

    def unexpected_main_agent():
        pytest.fail("Greetings must bypass the main agent")

    monkeypatch.setattr(graph_module, "build_main_agent", unexpected_main_agent)
    result = asyncio.run(
        build_graph().ainvoke({"messages": [HumanMessage(content="hello")]})
    )

    assert result["messages"][-1].content == "Hello. Welcome to interview me."


# Verifies graph state preserves the complete response emitted by a retrieval tool.
def test_graph_collects_requested_tool_responses(monkeypatch):
    class FakeMainAgent:
        async def astream(self, payload, context, config, stream_mode):
            assert context["name"]
            callbacks = config["callbacks"]
            handlers = getattr(callbacks, "handlers", callbacks)
            callback = next(
                handler
                for handler in handlers
                if handler.__class__.__name__ == "_ToolResponseCollector"
            )
            run_id = __import__("uuid").uuid4()
            callback.on_tool_start(
                {"name": "get_skills"}, "", run_id=run_id
            )
            callback.on_tool_end(
                "## Skills\n"
                "**Source:** `resume:skills`\n\n"
                "Python, LangChain",
                run_id=run_id,
            )
            yield AIMessage(content="I use Python."), {}

    monkeypatch.setattr(graph_module, "classify_with_model", _mock_guard)
    monkeypatch.setattr(graph_module, "build_main_agent", lambda: FakeMainAgent())

    result = asyncio.run(
        build_graph().ainvoke({"messages": [HumanMessage(content="What is your name?")]})
    )

    assert result["context"] == [
        "## Skills\n**Source:** `resume:skills`\n\nPython, LangChain"
    ]


def test_graph_collects_project_response_without_transforming_it(monkeypatch):
    tool_response = (
        "# Project evidence\n\n"
        "## 1. One\n\n"
        "- **Source:** `project:One`\n\n"
        "First project\n\n"
        "## 2. Two\n\n"
        "- **Source:** `project:Two`\n\n"
        "Second project"
    )

    class FakeMainAgent:
        async def astream(self, payload, context, config, stream_mode):
            assert context["name"]
            callback = config["callbacks"][0]
            run_id = __import__("uuid").uuid4()
            callback.on_tool_start(
                {"name": "get_project_detail"}, "", run_id=run_id
            )
            callback.on_tool_end(tool_response, run_id=run_id)
            yield AIMessage(content="I found two projects."), {}

    monkeypatch.setattr(graph_module, "classify_with_model", _mock_guard)
    monkeypatch.setattr(graph_module, "build_main_agent", lambda: FakeMainAgent())

    result = asyncio.run(
        build_graph().ainvoke(
            {"messages": [HumanMessage(content="Tell me about projects")]}
        )
    )

    assert result["context"] == [tool_response]


# Verifies guard categories remain constrained by constructing a valid decision directly.
def test_guard_decision_has_closed_categories():
    assert GuardDecision(category="safe").category == "safe"


def test_type_safe_guard_selects_highest_probability(monkeypatch):
    captured = {}

    class FakeClassifier:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

        def invoke(self, request, config=None):
            captured["request"] = request
            return SimpleNamespace(
                answers={
                    "category": SimpleNamespace(choice="greeting"),
                }
            )

    monkeypatch.setattr("src.backend.graph.decisions.TypeSafeClassifier", FakeClassifier)
    assert classify_with_model("hello") == GuardDecision(category="greeting")
    assert set(captured["request"]["questions"]) == {"category"}
    criteria = captured["request"]["questions"]["category"].criteria
    assert "answer may not exist" in criteria["safe"]
    assert "off-topic" in criteria["prohibited"]


# Verifies unavailable guard providers surface a runtime error by replacing model initialization with a failure.
def test_unavailable_guard_model_raises(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ConnectionError("provider unavailable")

    monkeypatch.setattr("src.backend.graph.decisions.TypeSafeClassifier", unavailable)
    with pytest.raises(RuntimeError, match="Guard model is unavailable"):
        from src.backend.graph.decisions import classify_with_model

        classify_with_model("Tell me about your experience")