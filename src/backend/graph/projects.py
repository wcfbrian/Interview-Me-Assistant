import heapq
import re
from concurrent.futures import ThreadPoolExecutor

from langchain.tools import tool

from src.backend.artifacts import load_projects
from src.backend.config import settings
from src.backend.ingest import get_project_bm25, get_project_vector_store
from src.backend.ingest.ingest_project import split_project_readme


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
    projects: list[dict[str, str]],
    candidate_count: int,
):
    bm25_index = get_project_bm25()

    # 1. Validate that the persisted BM25 index matches the current README chunks.
    chunks = [
        chunk
        for project in projects
        for chunk in split_project_readme(project)
    ]
    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    if (
        not isinstance(bm25_index, dict)
        or "bm25" not in bm25_index
        or bm25_index.get("chunk_ids") != chunk_ids
        or bm25_index["bm25"].corpus_size != len(chunks)
    ):
        raise RuntimeError(
            "Persisted BM25 project index is missing; "
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

    return [chunk_ids[index] for index, score in ranked if score > 0]


def _hybrid_project_search(query: str, projects: list[dict[str, str]], vector_store, k: int = 8):
    """Return the top README chunks from hybrid retrieval.

    Chroma and BM25 rank README chunks. Reciprocal rank fusion combines chunk
    rankings, allowing multiple ranked chunks to come from the same project.

    Args:
        query: Search text used by both retrieval methods.
        projects: Projects used to validate lexical IDs and resolve valid results.
        vector_store: Chroma vector store used for semantic retrieval.
        k: Maximum number of ranked chunks to return.

    Returns:
        A list of ``(project_id, rrf_score, passage, headers)`` tuples in
        descending score order.
    """

    with ThreadPoolExecutor(max_workers=2) as executor:
        semantic_future = executor.submit(
            _semantic_project_search, query, vector_store, k
        )
        lexical_future = executor.submit(
            _lexical_project_search, query, projects, k
        )
        vector_matches = semantic_future.result()
        lexical_order = lexical_future.result()
    projects_by_id = {
        str(project.get("project_id", "")): project
        for project in projects
    }
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

    chunks_by_id = {
        chunk.metadata["chunk_id"]: chunk
        for project in projects
        for chunk in split_project_readme(project)
    }
    chunk_scores = {
        chunk_id: score
        for chunk_id, score in ranked_chunks.items()
        if chunk_id in chunks_by_id
        and str(chunks_by_id[chunk_id].metadata["project_id"]) in projects_by_id
    }

    normalized_query = query.lower()
    for project in projects:
        project_id = str(project.get("project_id", ""))
        project_name = str(project.get("name", "")).strip()
        if project_name and project_name.lower() in normalized_query:
            project_chunk_ids = [
                chunk_id
                for chunk_id, chunk in chunks_by_id.items()
                if str(chunk.metadata["project_id"]) == project_id
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
        chunk = chunks_by_id[chunk_id]
        headers = {
            key: str(chunk.metadata[key])
            for key in ("h1", "h2")
            if chunk.metadata.get(key)
        }
        results.append(
            (
                str(chunk.metadata["project_id"]),
                score,
                chunk.page_content,
                headers,
            )
        )
    return results


@tool
def get_project_detail(query: str) -> str:
    """Return ranked project evidence as Markdown with source and section labels."""
    projects = load_projects(settings.project_path)
    projects_by_id = {
        str(project.get("project_id", "")): project
        for project in projects
    }
    matches = _hybrid_project_search(query, projects, get_project_vector_store())
    if not matches:
        return "# Project evidence\n\nNo relevant project evidence was found."

    sections = []
    for rank, (project_id, _, passage, headers) in enumerate(matches, start=1):
        project_name = str(projects_by_id[project_id].get("name", "unknown"))
        source = f"project:{project_name}"
        section = " > ".join(headers.values())
        lines = [
            f"## {rank}. {project_name}",
            "",
            f"- **Source:** `{source}`",
        ]
        if section:
            lines.append(f"- **Section:** {section}")
        lines.extend(["", passage.strip()])
        sections.append("\n".join(lines))

    return "# Project evidence\n\n" + "\n\n".join(sections)