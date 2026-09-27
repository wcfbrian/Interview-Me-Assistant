from langchain_core.messages import AIMessage
from rank_bm25 import BM25Okapi

from interview_me.backend.agents.projects import (
    _hybrid_project_search,
    _tokens,
    get_project_detail,
)


# Verifies RRF combines semantic and lexical candidates by running fusion with a fake vector store and BM25 index.
def test_hybrid_project_search_combines_vector_and_lexical_results(monkeypatch):
    projects = [
        {
            "project_id": "github-1",
            "name": "semantic",
            "summary": "retrieval system",
            "readme": "semantic",
        },
        {
            "project_id": "github-2",
            "name": "lexical",
            "summary": "BM25 keyword search",
            "readme": "lexical",
        },
    ]

    class FakeVectorStore:
        def similarity_search_with_score(self, query, k):
            from langchain_core.documents import Document

            return [
                (Document(page_content="semantic", metadata={"project_id": "github-1"}), 0.9)
            ]

    tokenized_corpus = [
        _tokens(f"{project['name']} {project['readme']}")
        for project in projects
    ]
    monkeypatch.setattr(
        "interview_me.backend.agents.projects.get_project_bm25",
        lambda: {
            "bm25": BM25Okapi(tokenized_corpus),
            "project_ids": [project["project_id"] for project in projects],
        },
    )

    matches = _hybrid_project_search("BM25 keyword", projects, FakeVectorStore(), k=2)

    assert [project_id for project_id, _ in matches] == ["github-1", "github-2"]
    assert all(score > 0 for _, score in matches)


# Verifies the public project tool delegates to its subagent by substituting an agent that asserts the invocation payload.
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