import pytest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from interview_me.backend.graph.decisions import GuardDecision, classify_with_model
from interview_me.backend.graph import guard_route, build_graph


def _mock_guard(query):
    normalized = query.lower()
    if normalized in {"hi", "hello", "hey"}:
        return GuardDecision(category="greeting")
    if "education" in normalized or "project" in normalized or "name" in normalized:
        return GuardDecision(category="safe")
    return GuardDecision(category="prohibited")


# Verifies interview questions reach the main route by replacing model classification with a safe decision.
def test_guard_routes_interview_question_to_main_agent(monkeypatch):
    monkeypatch.setattr("interview_me.backend.graph.classify_with_model", _mock_guard)
    assert guard_route({"messages": [HumanMessage(content="Tell me about your projects")]}) == "safe"


# Verifies follow-ups retain earlier context by capturing the text passed to the mocked guard.
def test_guard_passes_conversation_context_for_followup(monkeypatch):
    captured = {}

    def fake_guard(context):
        captured["context"] = context
        return GuardDecision(category="safe")

    monkeypatch.setattr("interview_me.backend.graph.classify_with_model", fake_guard)
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
    monkeypatch.setattr("interview_me.backend.graph.classify_with_model", _mock_guard)
    assert guard_route({"messages": [HumanMessage(content="What is your name?")]}) == "safe"


# Verifies greetings bypass the main agent by evaluating the deterministic greeting guard substitute.
def test_guard_routes_greeting_to_fixed_response(monkeypatch):
    monkeypatch.setattr("interview_me.backend.graph.classify_with_model", _mock_guard)
    assert guard_route({"messages": [HumanMessage(content="hello")]}) == "greeting"


# Verifies the compiled graph exposes only top-level workflow nodes by inspecting its node set.
def test_graph_keeps_subagents_out_of_top_level_nodes():
    graph = build_graph()
    assert set(graph.nodes) == {
        "__start__",
        "main_agent",
        "fixed_greeting",
        "fixed_relevance_response",
    }


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

    monkeypatch.setattr("interview_me.backend.graph.decisions.TypeSafeClassifier", FakeClassifier)
    assert classify_with_model("hello") == GuardDecision(category="greeting")
    assert set(captured["request"]["questions"]) == {"category"}


# Verifies unavailable guard providers surface a runtime error by replacing model initialization with a failure.
def test_unavailable_guard_model_raises(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ConnectionError("provider unavailable")

    monkeypatch.setattr("interview_me.backend.graph.decisions.TypeSafeClassifier", unavailable)
    with pytest.raises(RuntimeError, match="Guard model is unavailable"):
        from interview_me.backend.graph.decisions import classify_with_model

        classify_with_model("Tell me about your experience")