import pytest

from langchain_core.messages import HumanMessage

from interview_me.backend.artifacts import load_profile, load_projects
from interview_me.backend.decisions import AnswerDecision, GuardDecision


def _mock_guard(query):
    normalized = query.lower()
    if normalized in {"hi", "hello", "hey"}:
        return GuardDecision(category="greeting", reason="greeting")
    if "education" in normalized or "project" in normalized:
        return GuardDecision(category="relevant", reason="candidate question")
    return GuardDecision(category="irrelevant", reason="not a candidate question")
from interview_me.backend.graph import guard_route, build_graph


def test_guard_routes_interview_question_to_main_agent(monkeypatch):
    monkeypatch.setattr("interview_me.backend.graph.classify_with_model", _mock_guard)
    assert guard_route({"messages": [HumanMessage(content="Tell me about your projects")]}) == "relevant"


def test_guard_routes_greeting_to_fixed_response(monkeypatch):
    monkeypatch.setattr("interview_me.backend.graph.classify_with_model", _mock_guard)
    assert guard_route({"messages": [HumanMessage(content="hello")]}) == "greeting"


def test_graph_keeps_subagents_out_of_top_level_nodes():
    graph = build_graph()
    assert set(graph.nodes) == {
        "__start__",
        "main_agent",
        "fixed_greeting",
        "fixed_relevance_response",
        "fixed_abstention",
    }


def test_artifacts_preserve_only_json_profile_and_projects(tmp_path):
    profile_path = tmp_path / "profile.json"
    projects_path = tmp_path / "projects.json"
    profile_path.write_text('{"skills": "Python"}', encoding="utf-8")
    projects_path.write_text('[{"name": "demo", "readme": "RAG"}]', encoding="utf-8")

    assert load_profile(profile_path) == {"skills": "Python"}
    assert load_projects(projects_path)[0]["name"] == "demo"


def test_decision_schemas_have_closed_categories():
    assert GuardDecision(category="relevant", reason="candidate question").category == "relevant"
    assert AnswerDecision(approved=True, grounded=True, responsive=True).unsupported_claims == []


def test_unavailable_guard_model_raises(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ConnectionError("provider unavailable")

    monkeypatch.setattr("interview_me.backend.decisions.init_chat_model", unavailable)
    with pytest.raises(RuntimeError, match="Guard model is unavailable"):
        from interview_me.backend.decisions import classify_with_model

        classify_with_model("Tell me about your experience")