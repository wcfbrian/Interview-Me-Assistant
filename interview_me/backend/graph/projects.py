import heapq
import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from interview_me.backend.artifacts import load_projects
from interview_me.backend.config import settings, tracing_config
from interview_me.backend.ingest import get_project_bm25, get_project_vector_store


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _semantic_project_search(query: str, vector_store, candidate_count: int):
    return vector_store.similarity_search_with_score(query, k=candidate_count)


def _lexical_project_search(
    query: str,
    projects: list[dict[str, str]],
    candidate_count: int,
):
    bm25_index = get_project_bm25()

    # 1. Validate that the persisted BM25 index matches the current projects.
    project_ids = [str(project.get("project_id", "")) for project in projects]
    if (
        not isinstance(bm25_index, dict)
        or "bm25" not in bm25_index
        or bm25_index.get("project_ids") != project_ids
        or bm25_index["bm25"].corpus_size != len(projects)
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

    return [project_ids[index] for index, score in ranked]


def _hybrid_project_search(query: str, projects: list[dict[str, str]], vector_store, k: int = 3):
    """Return the top projects by combining semantic and lexical rankings.

    Chroma and BM25 each provide candidate IDs. Reciprocal rank fusion combines
    their rankings so projects found by both search methods receive contributions
    from both lists.

    Args:
        query: Search text used by both retrieval methods.
        projects: Projects used to validate lexical IDs and resolve valid results.
        vector_store: Chroma vector store used for semantic retrieval.
        k: Maximum number of ranked project results to return.

    Returns:
        A list of ``(project_id, rrf_score)`` tuples in descending score order.
    """
    candidate_count = max(k * 2, 6)

    with ThreadPoolExecutor(max_workers=2) as executor:
        semantic_future = executor.submit(
            _semantic_project_search, query, vector_store, candidate_count
        )
        lexical_future = executor.submit(
            _lexical_project_search, query, projects, candidate_count
        )
        vector_matches = semantic_future.result()
        lexical_order = lexical_future.result()
    projects_by_id = {
        str(project.get("project_id", "")): project
        for project in projects
    }
    ranked: dict[int, dict] = {}
    fusion_constant = 60
    semantic_weight = 1.0
    lexical_weight = 1.0
    query_tokens = set(_tokens(query))

    for rank, (document, _) in enumerate(vector_matches, start=1):
        project_id = str(document.metadata.get("project_id", ""))
        ranked.setdefault(project_id, {"rrf_score": 0.0})["rrf_score"] += (
            semantic_weight / (fusion_constant + rank)
        )

    for rank, project_id in enumerate(lexical_order, start=1):
        if project_id in projects_by_id:
            ranked.setdefault(project_id, {"rrf_score": 0.0})["rrf_score"] += (
                lexical_weight / (fusion_constant + rank)
            )

    for project_id, project in projects_by_id.items():
        name_tokens = set(_tokens(project.get("name", "")))
        if name_tokens and name_tokens <= query_tokens:
            ranked.setdefault(project_id, {"rrf_score": 0.0})["rrf_score"] += 1.0

    return [
        (project_id, values["rrf_score"])
        for project_id, values in heapq.nlargest(
            k, ranked.items(), key=lambda item: item[1]["rrf_score"]
        )
        if project_id in projects_by_id
    ]


@tool
def _retrieve_project_evidence(query: str) -> str:
    """Return project evidence from fused semantic and lexical retrieval."""
    projects = load_projects(settings.project_path)
    projects_by_id = {
        str(project.get("project_id", "")): project
        for project in projects
    }
    matches = _hybrid_project_search(query, projects, get_project_vector_store())
    evidence = [
        {
            "source": f"project:{projects_by_id[project_id].get('name', 'unknown')}",
            "text": projects_by_id[project_id].get("readme") or projects_by_id[project_id].get("summary", ""),
            "score": score,
        }
        for project_id, score in matches
    ]
    return json.dumps({"status": "found" if evidence else "not_found", "evidence": evidence})


@lru_cache(maxsize=1)
def build_project_agent():
    """Build the project subagent that owns semantic project retrieval."""
    model = init_chat_model(settings.main_model, model_provider="openrouter")
    return create_agent(
        model=model,
        tools=[_retrieve_project_evidence],
        system_prompt=(
            "You are the project evidence subagent. Always call the retrieval tool. "
            "Return related content only. Do not invent, or add facts."
        ),
    )


@tool
def get_project_detail(query: str) -> str:
    """Delegate project questions to the semantic project evidence subagent."""
    response = build_project_agent().invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=tracing_config("interview-project-agent", "project-agent"),
    )
    return str(response["messages"][-1].content)