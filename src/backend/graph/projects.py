import heapq
import re
from concurrent.futures import ThreadPoolExecutor

from langchain.tools import tool

from src.backend.ingest import get_project_bm25, get_project_vector_store


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _semantic_project_search(query: str, vector_store, candidate_count: int):
    return [
        (document, score)
        for document, score in vector_store.similarity_search_with_score(
            query, k=candidate_count
        )
        if score > 0
    ]


def _lexical_project_search(
    query: str,
    candidate_count: int,
):
    bm25_index = get_project_bm25()
    chunk_ids = bm25_index.get("chunk_ids") if isinstance(bm25_index, dict) else None
    if (
        not isinstance(bm25_index, dict)
        or "bm25" not in bm25_index
        or not isinstance(chunk_ids, list)
        or not isinstance(bm25_index.get("project_names"), list)
        or len(bm25_index["project_names"]) != len(chunk_ids)
        or bm25_index["bm25"].corpus_size != len(chunk_ids)
    ):
        raise RuntimeError(
            "Persisted BM25 project index is missing or incompatible; "
            "run project ingestion before searching."
        )

    # 2. Score every project, then retain only the highest-ranked candidates.
    bm25 = bm25_index["bm25"]
    scores = bm25.get_scores(list(set(_tokens(query))))
    ranked = heapq.nlargest(
        candidate_count,
        enumerate(scores),
        key=lambda x: x[1],  # Sort by score
    )

    ranked_ids = [chunk_ids[index] for index, score in ranked if score > 0]
    matching_names = {
        str(name).strip().lower()
        for name in bm25_index["project_names"]
        if str(name).strip() and str(name).strip().lower() in query.lower()
    }
    exact_match_ids = [
        chunk_id
        for chunk_id, name in zip(chunk_ids, bm25_index["project_names"])
        if str(name).strip().lower() in matching_names
    ]
    return list(dict.fromkeys([*ranked_ids, *exact_match_ids]))


def _hybrid_project_search(query: str, vector_store, k: int = 8):
    """Return the top README chunks from hybrid retrieval.

    Chroma and BM25 rank README chunks. Reciprocal rank fusion combines chunk
    rankings, and Chroma supplies the matching documents and metadata.

    Args:
        query: Search text used by both retrieval methods.
        vector_store: Chroma vector store used for semantic retrieval.
        k: Maximum number of ranked chunks to return.

    Returns:
        A list of ``(document, rrf_score)`` tuples in descending score order.
    """

    with ThreadPoolExecutor(max_workers=2) as executor:
        semantic_future = executor.submit(
            _semantic_project_search, query, vector_store, k
        )
        lexical_future = executor.submit(
            _lexical_project_search, query, k
        )
        vector_matches = semantic_future.result()
        lexical_order = lexical_future.result()

    documents_by_id = {
        str(document.metadata["chunk_id"]): document
        for document, _ in vector_matches
        if document.metadata.get("chunk_id")
    }
    lexical_ids_to_fetch = [
        chunk_id for chunk_id in lexical_order if chunk_id not in documents_by_id
    ]
    if lexical_ids_to_fetch:
        for document in vector_store.get_by_ids(lexical_ids_to_fetch):
            chunk_id = str(document.metadata.get("chunk_id", ""))
            if chunk_id:
                documents_by_id[chunk_id] = document
    missing_lexical_ids = [
        chunk_id for chunk_id in lexical_order if chunk_id not in documents_by_id
    ]
    if missing_lexical_ids:
        raise RuntimeError(
            "Persisted BM25 project index does not match the Chroma project index; "
            "run project ingestion before searching."
        )

    ranked_chunks: dict[str, float] = {}
    fusion_constant = 60
    semantic_weight = 1.0
    lexical_weight = 1.0

    for rank, (document, _) in enumerate(vector_matches, start=1):
        chunk_id = str(document.metadata.get("chunk_id", ""))
        if chunk_id not in ranked_chunks:
            ranked_chunks[chunk_id] = 0.0
        ranked_chunks[chunk_id] += semantic_weight / (fusion_constant + rank)

    for rank, chunk_id in enumerate(lexical_order, start=1):
        if chunk_id not in ranked_chunks:
            ranked_chunks[chunk_id] = 0.0
        ranked_chunks[chunk_id] += lexical_weight / (fusion_constant + rank)

    chunk_scores = {
        chunk_id: score
        for chunk_id, score in ranked_chunks.items()
        if chunk_id in documents_by_id
    }

    normalized_query = query.lower()
    matching_project_ids = {
        str(document.metadata.get("project_id", ""))
        for document in documents_by_id.values()
        if str(document.metadata.get("name", "")).strip().lower() in normalized_query
        and document.metadata.get("project_id")
    }
    for project_id in matching_project_ids:
        project_chunk_ids = [
            chunk_id
            for chunk_id, document in documents_by_id.items()
            if str(document.metadata.get("project_id", "")) == project_id
        ]
        if project_chunk_ids:
            best_chunk_id = max(
                project_chunk_ids,
                key=lambda chunk_id: chunk_scores.get(chunk_id, 0.0),
            )
            chunk_scores[best_chunk_id] = max(
                chunk_scores.get(best_chunk_id, 0.0),
                1.0,
            )

    results = []
    for chunk_id, score in heapq.nlargest(
        k, chunk_scores.items(), key=lambda item: item[1]
    ):
        results.append((documents_by_id[chunk_id], score))
    return results


@tool
def get_project_detail(query: str) -> str:
    """Return ranked project evidence as Markdown with source and section labels."""
    vector_store = get_project_vector_store()
    matches = _hybrid_project_search(query, vector_store)
    if not matches:
        return "# Project evidence\n\nNo relevant project evidence was found."

    sections = []
    for rank, (document, _) in enumerate(matches, start=1):
        project_name = str(document.metadata.get("name", "unknown"))
        source = f"project:{project_name}"
        headers = {
            key: str(document.metadata[key])
            for key in ("h1", "h2")
            if document.metadata.get(key)
        }
        section = " > ".join(headers.values())
        lines = [
            f"## {rank}. {project_name}",
            "",
            f"- **Source:** `{source}`",
        ]
        if section:
            lines.append(f"- **Section:** {section}")
        lines.extend(["\n", document.page_content.strip()])
        sections.append("\n".join(lines))

    return "# Project evidence\n\n" + "\n\n".join(sections)