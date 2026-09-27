import json
from types import SimpleNamespace

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from interview_me.backend.artifacts import load_profile, load_projects
from interview_me.backend.decisions import GuardDecision
from interview_me.backend.ingest import ingest
from interview_me.backend.ingest.ingest_project import ProjectSummary, index_projects
from interview_me.backend.ingest.ingest_resume import CandidateProfile
from interview_me.backend.agents.projects import (
    _hybrid_project_search,
    _tokens,
    get_project_detail,
)
from rank_bm25 import BM25Okapi


def _mock_guard(query):
    normalized = query.lower()
    if normalized in {"hi", "hello", "hey"}:
        return GuardDecision(category="greeting", reason="greeting")
    if "education" in normalized or "project" in normalized or "name" in normalized:
        return GuardDecision(category="safe", reason="candidate question")
    return GuardDecision(category="prohibited", reason="not a candidate question")
from interview_me.backend.graph import guard_route, build_graph


def test_guard_routes_interview_question_to_main_agent(monkeypatch):
    monkeypatch.setattr("interview_me.backend.graph.classify_with_model", _mock_guard)
    assert guard_route({"messages": [HumanMessage(content="Tell me about your projects")]}) == "safe"


def test_guard_passes_conversation_context_for_followup(monkeypatch):
    captured = {}

    def fake_guard(context):
        captured["context"] = context
        return GuardDecision(category="safe", reason="follow-up")

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


def test_guard_allows_candidate_name_question(monkeypatch):
    monkeypatch.setattr("interview_me.backend.graph.classify_with_model", _mock_guard)
    assert guard_route({"messages": [HumanMessage(content="What is your name?")]}) == "safe"


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
    }


def test_artifacts_preserve_only_json_profile_and_projects(tmp_path):
    profile_path = tmp_path / "profile.json"
    projects_path = tmp_path / "projects.json"
    profile_path.write_text('{"skills": "Python"}', encoding="utf-8")
    projects_path.write_text('[{"name": "demo", "readme": "RAG"}]', encoding="utf-8")

    assert load_profile(profile_path) == {"skills": "Python"}
    assert load_projects(projects_path)[0]["name"] == "demo"


def test_guard_decision_has_closed_categories():
    assert GuardDecision(category="safe", reason="candidate question").category == "safe"


def test_project_summary_allows_fewer_than_300_words():
    assert len(ProjectSummary(summary="word " * 299).summary.split()) == 299


def test_project_summary_rejects_300_words():
    with pytest.raises(ValueError, match="fewer than 300 words"):
        ProjectSummary(summary="word " * 300)


def test_ingest_generates_missing_project_index(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    project_path = output_dir / "projects.json"
    profile = CandidateProfile(
        self_introduction="I build software.",
        working_background="Engineering",
        education="Computer science",
        skills="Python",
        highlighted_project="A project",
    )
    monkeypatch.setattr("interview_me.backend.ingest.extract_resume_text", lambda _: "resume")
    monkeypatch.setattr("interview_me.backend.ingest.extract_profile", lambda _: profile)
    monkeypatch.setattr(
        "interview_me.backend.ingest.settings",
        SimpleNamespace(github_username="candidate"),
    )

    def fake_sync(username, output, regenerate_json=False, token=""):
        assert username == "candidate"
        output.write_text('[{"name": "demo"}]', encoding="utf-8")
        return 1

    monkeypatch.setattr("interview_me.backend.ingest.pipeline", fake_sync)

    ingest(tmp_path / "resume.pdf", output_dir)
    assert json.loads((output_dir / "projects.json").read_text()) == [{"name": "demo"}]


def test_ingest_regenerates_existing_project_index(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    project_path = output_dir / "projects.json"
    project_path.write_text('[{"name": "old"}]', encoding="utf-8")
    profile = CandidateProfile(
        self_introduction="I build software.",
        working_background="Engineering",
        education="Computer science",
        skills="Python",
        highlighted_project="A project",
    )
    monkeypatch.setattr("interview_me.backend.ingest.extract_resume_text", lambda _: "resume")
    monkeypatch.setattr("interview_me.backend.ingest.extract_profile", lambda _: profile)
    monkeypatch.setattr(
        "interview_me.backend.ingest.settings",
        SimpleNamespace(github_username="candidate"),
    )

    def fake_sync(username, output, regenerate_json=False, token=""):
        output.write_text('[{"name": "new"}]', encoding="utf-8")
        return 1

    monkeypatch.setattr("interview_me.backend.ingest.pipeline", fake_sync)

    ingest(tmp_path / "resume.pdf", output_dir, regenerate_json=True)
    assert json.loads(project_path.read_text()) == [{"name": "new"}]


def test_index_projects_embeds_summaries_with_project_metadata(monkeypatch, tmp_path):
    project_path = tmp_path / "projects.json"
    project_path.write_text(
        json.dumps([{
            "name": "demo",
            "summary": "A retrieval project",
            "readme": "Full approved README",
            "source_url": "https://github.com/example/demo",
        }]),
        encoding="utf-8",
    )

    class FakeVectorStore:
        def add_documents(self, documents, ids):
            self.documents = documents
            self.ids = ids

    vector_store = FakeVectorStore()
    monkeypatch.setattr(
        "interview_me.backend.ingest.ingest_project.reset_project_vector_store",
        lambda: vector_store,
    )

    assert index_projects(project_path) == 1
    assert vector_store.documents[0].page_content == "A retrieval project"
    assert vector_store.documents[0].metadata["name"] == "demo"
    assert vector_store.ids == ["project-0"]


def test_hybrid_project_search_combines_vector_and_lexical_results(monkeypatch):
    projects = [
        {"name": "semantic", "summary": "retrieval system", "readme": "semantic"},
        {"name": "lexical", "summary": "BM25 keyword search", "readme": "lexical"},
    ]

    class FakeVectorStore:
        def similarity_search_with_score(self, query, k):
            from langchain_core.documents import Document

            return [(Document(page_content="semantic", metadata={"name": "semantic"}), 0.9)]

    tokenized_corpus = [
        _tokens(f"{project['name']} {project['readme']}")
        for project in projects
    ]
    monkeypatch.setattr(
        "interview_me.backend.agents.projects.get_project_bm25",
        lambda: {
            "bm25": BM25Okapi(tokenized_corpus),
            "tokenized_corpus": tokenized_corpus,
        },
    )
    matches = _hybrid_project_search("BM25 keyword", projects, FakeVectorStore(), k=2)

    assert [project["name"] for project, _ in matches] == ["semantic", "lexical"]


def test_project_tool_delegates_to_project_subagent(monkeypatch):
    class FakeProjectAgent:
        def invoke(self, payload, **kwargs):
            assert payload["messages"][0]["content"] == "Tell me about RAG"
            assert kwargs["config"]["run_name"] == "interview-project-agent"
            return {"messages": [AIMessage(content='{"status": "found"}')]} 

    monkeypatch.setattr(
        "interview_me.backend.agents.projects.build_project_agent",
        lambda: FakeProjectAgent(),
    )

    assert get_project_detail.invoke({"query": "Tell me about RAG"}) == '{"status": "found"}'


def test_unavailable_guard_model_raises(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ConnectionError("provider unavailable")

    monkeypatch.setattr("interview_me.backend.decisions.init_chat_model", unavailable)
    with pytest.raises(RuntimeError, match="Guard model is unavailable"):
        from interview_me.backend.decisions import classify_with_model

        classify_with_model("Tell me about your experience")