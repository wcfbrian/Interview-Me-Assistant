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
    bm25_index: dict | None = None,
):
    query_tokens = set(_tokens(query))
    if (
        not isinstance(bm25_index, dict)
        or "bm25" not in bm25_index
        or "tokenized_corpus" not in bm25_index
        or bm25_index["bm25"].corpus_size != len(projects)
    ):
        raise RuntimeError(
            "Persisted BM25 project index is missing; "
            "run project ingestion before searching."
        )
    bm25 = bm25_index["bm25"]
    tokenized_corpus = bm25_index["tokenized_corpus"]
    lexical_scores = bm25.get_scores(list(query_tokens))
    lexical_order = sorted(
        range(len(projects)),
        key=lambda index: (
            lexical_scores[index],
            len(query_tokens.intersection(tokenized_corpus[index])),
        ),
        reverse=True,
    )
    lexical_matches = [
        index
        for index in lexical_order
        if lexical_scores[index] > 0
        or query_tokens.intersection(tokenized_corpus[index])
    ]
    return (lexical_matches or lexical_order)[:candidate_count]


def _hybrid_project_search(query: str, projects: list[dict[str, str]], vector_store, k: int = 3):
    """Fuse Chroma semantic ranks with BM25 lexical ranks using reciprocal rank fusion."""
    candidate_count = max(k * 2, 6)
    with ThreadPoolExecutor(max_workers=2) as executor:
        semantic_future = executor.submit(
            _semantic_project_search, query, vector_store, candidate_count
        )
        lexical_future = executor.submit(
            _lexical_project_search, query, projects, candidate_count, get_project_bm25()
        )
        vector_matches = semantic_future.result()
        lexical_order = lexical_future.result()
    project_indexes = {
        str(project.get("name", "unknown")): index
        for index, project in enumerate(projects)
    }
    ranked: dict[int, dict] = {}
    fusion_constant = 60
    semantic_weight = 1.0
    lexical_weight = 1.0

    for rank, (document, _) in enumerate(vector_matches, start=1):
        index = project_indexes.get(str(document.metadata.get("name", "")))
        if index is not None:
            ranked.setdefault(index, {"rrf_score": 0.0})["rrf_score"] += (
                semantic_weight / (fusion_constant + rank)
            )

    for rank, index in enumerate(lexical_order, start=1):
        ranked.setdefault(index, {"rrf_score": 0.0})["rrf_score"] += (
            lexical_weight / (fusion_constant + rank)
        )

    return [
        (projects[index], values["rrf_score"])
        for index, values in sorted(
            ranked.items(), key=lambda item: item[1]["rrf_score"], reverse=True
        )[:k]
    ]


@tool
def _retrieve_project_evidence(query: str) -> str:
    """Return project evidence from fused semantic and lexical retrieval."""
    projects = load_projects(settings.project_path)
    matches = _hybrid_project_search(query, projects, get_project_vector_store())
    evidence = [
        {
            "source": f"project:{project.get('name', 'unknown')}",
            "text": project.get("readme") or project.get("summary", ""),
            "score": score,
        }
        for project, score in matches
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