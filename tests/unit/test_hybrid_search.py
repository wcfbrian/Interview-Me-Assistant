from backend.retrieval.hybrid import HybridRetriever, SearchDocument


def test_hybrid_search_prioritizes_matching_project_metadata() -> None:
    retriever = HybridRetriever(
        [
            SearchDocument("rag", "Hybrid RAG with BM25 and vector search", "rag.md"),
            SearchDocument("vision", "Image classification pipeline", "vision.md"),
        ]
    )

    results = retriever.search("What RAG project used hybrid search?", limit=1)

    assert len(results) == 1
    assert results[0].document.name == "rag"


def test_hybrid_search_returns_empty_for_no_documents() -> None:
    assert HybridRetriever([]).search("any query") == []