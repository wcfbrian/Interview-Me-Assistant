from rank_bm25 import BM25Okapi

from src.backend.graph.projects import (
    _hybrid_project_search,
    _lexical_project_search,
    _semantic_project_search,
    _tokens,
    get_project_detail,
)
from src.backend.ingest.ingest_project import split_project_readme


# Verifies RRF combines semantic and lexical candidates by running fusion with a fake vector store and BM25 index.
def test_hybrid_project_search_combines_vector_and_lexical_results(monkeypatch):
    projects = [
        {
            "project_id": "github-1",
            "name": "semantic",
            "readme": "# semantic\n\nretrieval system",
        },
        {
            "project_id": "github-2",
            "name": "lexical",
            "readme": "# lexical\n\nBM25 keyword lexical",
        },
        {
            "project_id": "github-3",
            "name": "unrelated",
            "readme": "# unrelated\n\nunrelated content",
        },
    ]

    class FakeVectorStore:
        def similarity_search_with_score(self, query, k):
            from langchain_core.documents import Document

            return [
                (
                    Document(
                        page_content="# semantic\n\nretrieval system",
                        metadata={
                            "project_id": "github-1",
                            "chunk_id": "github-1:chunk:0",
                        },
                    ),
                    0.9,
                )
            ]

    chunks = [
        (project, chunk)
        for project in projects
        for chunk in split_project_readme(project)
    ]
    tokenized_corpus = [
        _tokens(f"{project['name']} {chunk.page_content}")
        for project, chunk in chunks
    ]
    monkeypatch.setattr(
        "src.backend.graph.projects.get_project_bm25",
        lambda: {
            "bm25": BM25Okapi(tokenized_corpus),
            "chunk_ids": [chunk.metadata["chunk_id"] for _, chunk in chunks],
        },
    )

    matches = _hybrid_project_search("BM25 keyword", projects, FakeVectorStore(), k=2)

    assert [project_id for project_id, _, _, _ in matches] == ["github-1", "github-2"]
    assert all(score > 0 for _, score, _, _ in matches)
    assert matches[0][3] == {"h1": "semantic"}


def test_hybrid_project_search_defaults_to_top_eight(monkeypatch):
    projects = [
        {
            "project_id": f"github-{index}",
            "name": f"project-{index}",
            "readme": f"# project-{index}\n\nproject retrieval",
        }
        for index in range(10)
    ]
    chunks = [
        chunk
        for project in projects
        for chunk in split_project_readme(project)
    ]
    candidate_counts = []

    def semantic_search(query, vector_store, candidate_count):
        candidate_counts.append(candidate_count)
        return [(chunk, 1.0) for chunk in chunks]

    def lexical_search(query, loaded_projects, candidate_count):
        candidate_counts.append(candidate_count)
        return [chunk.metadata["chunk_id"] for chunk in chunks]

    monkeypatch.setattr(
        "src.backend.graph.projects._semantic_project_search", semantic_search
    )
    monkeypatch.setattr(
        "src.backend.graph.projects._lexical_project_search", lexical_search
    )

    matches = _hybrid_project_search("project retrieval", projects, object())

    assert len(matches) == 8
    assert candidate_counts == [8, 8]


def test_hybrid_project_search_keeps_multiple_chunks_from_same_project(monkeypatch):
    from langchain_core.documents import Document

    project = {
        "project_id": "github-qa",
        "name": "QA_bot",
        "readme": "",
    }
    chunks = [
        Document(
            page_content="First matching passage",
            metadata={
                "project_id": "github-qa",
                "chunk_id": "github-qa:chunk:0",
                "h1": "QA_bot",
                "h2": "Overview",
            },
        ),
        Document(
            page_content="Second matching passage",
            metadata={
                "project_id": "github-qa",
                "chunk_id": "github-qa:chunk:1",
                "h1": "QA_bot",
                "h2": "Retrieval",
            },
        ),
    ]
    monkeypatch.setattr(
        "src.backend.graph.projects.split_project_readme",
        lambda _: chunks,
    )
    monkeypatch.setattr(
        "src.backend.graph.projects._semantic_project_search",
        lambda query, vector_store, candidate_count: [
            (chunks[0], 1.0),
            (chunks[1], 0.9),
        ],
    )
    monkeypatch.setattr(
        "src.backend.graph.projects._lexical_project_search",
        lambda query, loaded_projects, candidate_count: [
            "github-qa:chunk:0",
            "github-qa:chunk:1",
        ],
    )

    matches = _hybrid_project_search("multimodal retrieval", [project], object(), k=2)

    assert [project_id for project_id, _, _, _ in matches] == [
        "github-qa",
        "github-qa",
    ]
    assert [passage for _, _, passage, _ in matches] == [
        "First matching passage",
        "Second matching passage",
    ]


def test_project_search_returns_empty_for_non_positive_scores(monkeypatch):
    projects = [
        {"project_id": "github-1", "name": "alpha", "readme": "# Alpha\n\nfirst project"},
        {"project_id": "github-2", "name": "beta", "readme": "# Beta\n\nsecond project"},
    ]

    class FakeVectorStore:
        def similarity_search_with_score(self, query, k):
            from langchain_core.documents import Document

            return [
                (
                    Document(
                        page_content="alpha",
                        metadata={
                            "project_id": "github-1",
                            "chunk_id": "github-1:chunk:0",
                        },
                    ),
                    0.0,
                ),
            ]

    chunks = [
        (project, chunk)
        for project in projects
        for chunk in split_project_readme(project)
    ]
    tokenized_corpus = [
        _tokens(f"{project['name']} {chunk.page_content}")
        for project, chunk in chunks
    ]
    monkeypatch.setattr(
        "src.backend.graph.projects.get_project_bm25",
        lambda: {
            "bm25": BM25Okapi(tokenized_corpus),
            "chunk_ids": [chunk.metadata["chunk_id"] for _, chunk in chunks],
        },
    )

    query = "missing technology"
    vector_store = FakeVectorStore()

    assert _semantic_project_search(query, vector_store, candidate_count=2) == []
    assert _lexical_project_search(query, projects, candidate_count=2) == []
    assert _hybrid_project_search(query, projects, vector_store, k=2) == []


def test_hybrid_project_search_prioritizes_explicit_project_name(monkeypatch):
    projects = [
        {"project_id": "github-1", "name": "other", "readme": "QA assistant"},
        {
            "project_id": "github-2",
            "name": "QA_bot",
            "readme": "# QA Bot\n\n## Usage\n\nhybrid retrieval",
        },
    ]

    class FakeVectorStore:
        def similarity_search_with_score(self, query, k):
            from langchain_core.documents import Document

            return [
                (Document(page_content="other", metadata={"project_id": "github-1"}), 0.9)
            ]

    chunks = [
        (project, chunk)
        for project in projects
        for chunk in split_project_readme(project)
    ]
    tokenized_corpus = [
        _tokens(f"{project['name']} {chunk.page_content}")
        for project, chunk in chunks
    ]
    monkeypatch.setattr(
        "src.backend.graph.projects.get_project_bm25",
        lambda: {
            "bm25": BM25Okapi(tokenized_corpus),
            "chunk_ids": [chunk.metadata["chunk_id"] for _, chunk in chunks],
        },
    )

    matches = _hybrid_project_search(
        "What did you build in QA_bot?", projects, FakeVectorStore(), k=2
    )

    assert matches[0][0] == "github-2"
    assert matches[0][3] == {"h1": "QA Bot", "h2": "Usage"}


def test_project_tool_returns_retrieved_evidence_directly(monkeypatch):
    projects = [
        {
            "project_id": "github-1",
            "name": "RAG",
            "readme": "# RAG\n\nHybrid project retrieval",
        }
    ]
    monkeypatch.setattr("src.backend.graph.projects.load_projects", lambda _: projects)
    monkeypatch.setattr("src.backend.graph.projects.get_project_vector_store", lambda: object())
    monkeypatch.setattr(
        "src.backend.graph.projects._hybrid_project_search",
        lambda query, loaded_projects, vector_store: [
            ("github-1", 1.0, "Hybrid project retrieval", {"h1": "RAG"})
        ],
    )

    result = get_project_detail.invoke({"query": "Tell me about RAG"})

    assert result == (
        "# Project evidence\n\n"
        "## 1. RAG\n\n"
        "- **Source:** `project:RAG`\n"
        "- **Section:** RAG\n\n"
        "Hybrid project retrieval"
    )


def test_project_tool_formats_eight_ranked_results_as_markdown(monkeypatch):
    projects = [
        {"project_id": f"github-{rank}", "name": f"Project {rank}"}
        for rank in range(1, 9)
    ]
    matches = [
        (
            f"github-{rank}",
            1 / rank,
            f"Evidence for project {rank}",
            {"h1": f"Project {rank}", "h2": "Overview"},
        )
        for rank in range(1, 9)
    ]
    monkeypatch.setattr("src.backend.graph.projects.load_projects", lambda _: projects)
    monkeypatch.setattr("src.backend.graph.projects.get_project_vector_store", lambda: object())
    monkeypatch.setattr(
        "src.backend.graph.projects._hybrid_project_search",
        lambda query, loaded_projects, vector_store: matches,
    )

    result = get_project_detail.invoke({"query": "Tell me about my projects"})

    assert result.count("\n\n## ") == 8
    for rank in range(1, 9):
        assert f"## {rank}. Project {rank}" in result
        assert f"- **Source:** `project:Project {rank}`" in result
        assert f"- **Section:** Project {rank} > Overview" in result
        assert f"Evidence for project {rank}" in result
    assert result.index("## 1. Project 1") < result.index("## 8. Project 8")


def test_project_tool_returns_markdown_when_no_evidence(monkeypatch):
    monkeypatch.setattr("src.backend.graph.projects.load_projects", lambda _: [])
    monkeypatch.setattr("src.backend.graph.projects.get_project_vector_store", lambda: object())
    monkeypatch.setattr(
        "src.backend.graph.projects._hybrid_project_search",
        lambda query, loaded_projects, vector_store: [],
    )

    result = get_project_detail.invoke({"query": "Tell me about an unknown project"})

    assert result == "# Project evidence\n\nNo relevant project evidence was found."